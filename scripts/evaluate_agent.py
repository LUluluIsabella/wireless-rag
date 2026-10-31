# 用途：读取Agent评测集，逐条执行真实问答并保存结果。
# 自动检查作答状态及要求的工具是否成功执行，汇总调用次数和耗时。
# 证据是否支持答案、工具调用是否必要，仍需人工审核。
# 每完成一条就保存一次，便于在运行中断后查看已有结果。
# 本脚本会调用真实模型API，产生相应费用。

import argparse
import json
from collections import Counter
from pathlib import Path

from src.agent import WifiAgent


def load_cases(path):
    # 读取JSONL，并在发送真实API请求前检查全部用例。
    cases = []
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第{line_number}行不是合法JSON：{exc}"
                ) from exc

            if not isinstance(case, dict):
                raise ValueError(f"第{line_number}行必须是JSON对象")

            case_id = case.get("id")
            question = case.get("question")

            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"第{line_number}行缺少有效id")

            if case_id in seen_ids:
                raise ValueError(f"存在重复id：{case_id}")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{case_id}缺少有效question")

            if "expected_answerable" not in case:
                raise ValueError(
                    f"{case_id}缺少expected_answerable"
                )

            expected = case["expected_answerable"]
            if expected is not None and type(expected) is not bool:
                raise ValueError(
                    f"{case_id}的expected_answerable必须为"
                    "true、false或null"
                )

            required = case.get("required_tools", [])
            if not isinstance(required, list) or any(
                not isinstance(name, str) or not name.strip()
                for name in required
            ):
                raise ValueError(
                    f"{case_id}的required_tools必须为工具名列表"
                )

            seen_ids.add(case_id)
            cases.append(case)

    if not cases:
        raise ValueError("评测集为空")

    return cases


def check_result(case, result):
    # 仅将明确回答和明确拒答用于预期状态比较。
    # 模型错误、超时等状态不能算作正确拒答。
    expected = case["expected_answerable"]
    status = result.get("status")

    if expected is None:
        status_matches = None
    else:
        expected_status = "answered" if expected else "abstained"
        status_matches = status == expected_status

    # 上一版trace中的tool字段保存工具名，error保存错误信息。
    # 只有没有错误记录的调用，才视为成功执行。
    successful_tools = {
        entry["tool"]
        for entry in result.get("trace", [])
        if entry.get("tool") and not entry.get("error")
    }

    required = set(case.get("required_tools", []))
    missing = sorted(required - successful_tools)

    return {
        "status_matches": status_matches,
        "successful_tools": sorted(successful_tools),
        "missing_required_tools": missing,
        # 空工具列表表示没有要求，汇总时不计入工具满足率。
        "required_tools_satisfied": (
            not missing if required else None
        ),
    }


def summarize(records):
    # 分别统计各指标的有效分母，避免将不适用项算作通过。
    status_checks = [
        row["checks"]["status_matches"]
        for row in records
        if row["checks"]["status_matches"] is not None
    ]
    tool_checks = [
        row["checks"]["required_tools_satisfied"]
        for row in records
        if row["checks"]["required_tools_satisfied"] is not None
    ]

    count = len(records)

    return {
        "completed_cases": count,
        "status_counts": dict(
            Counter(row["result"]["status"] for row in records)
        ),
        "status_match": {
            "passed": sum(status_checks),
            "evaluated": len(status_checks),
            "rate": (
                sum(status_checks) / len(status_checks)
                if status_checks else None
            ),
        },
        "required_tools": {
            "passed": sum(tool_checks),
            "evaluated": len(tool_checks),
            "rate": (
                sum(tool_checks) / len(tool_checks)
                if tool_checks else None
            ),
        },
        "average_tool_calls": (
            sum(row["result"].get("tool_calls", 0) for row in records)
            / count if count else 0
        ),
        "average_elapsed_seconds": (
            sum(
                row["result"].get("elapsed_seconds", 0)
                for row in records
            ) / count if count else 0
        ),
    }


def save_report(path, dataset, total_cases, records):
    # 保存完整轨迹和待审核字段，不只保留汇总分数。
    report = {
        "dataset": str(dataset),
        "total_cases": total_cases,
        "summary": summarize(records),
        "results": records,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="批量评估Wi-Fi Agent"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/agent_dev.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/agent_dev_results.json"),
    )
    args = parser.parse_args()

    # 防止误用相同路径覆盖评测集。
    if args.dataset.resolve() == args.output.resolve():
        parser.error("--output不能与--dataset使用同一路径")

    cases = load_cases(args.dataset)

    # 复用客户端与检索器；每次run内部会重新创建任务状态。
    agent = WifiAgent()
    records = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case['id']} 正在运行……",
            flush=True,
        )

        result = agent.run(case["question"])
        checks = check_result(case, result)

        records.append(
            {
                "case": case,
                "result": result,
                "checks": checks,
                "manual_review": {
                    "evidence_supports_answer": None,
                    "tool_use_reasonable": None,
                    "abstention_reasonable": None,
                    "notes": "",
                },
            }
        )

        report = save_report(
            args.output, args.dataset, len(cases), records
        )

        print(
            f"  status={result['status']} | "
            f"状态符合={checks['status_matches']} | "
            f"工具满足={checks['required_tools_satisfied']} | "
            f"调用次数={result.get('tool_calls', 0)}",
            flush=True,
        )

    print("\n=== 自动检查汇总 ===")
    print(
        json.dumps(
            report["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n结果已保存：{args.output}")


if __name__ == "__main__":
    main()
