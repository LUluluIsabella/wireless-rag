# 用途：从CLI运行Wi-Fi文档Agent。
# 展示工具动作、证据编号、最终答案和真实来源。
# 可选保存运行结果；不保存密钥或模型内部推理。

import argparse
import json
from pathlib import Path

from src.agent import WifiAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    agent = WifiAgent()
    result = agent.run(args.question)

    print("\n=== 工具轨迹 ===")

    for step in result["trace"]:
        print(json.dumps(step, ensure_ascii=False, indent=2))

    print("\n=== 最终答案 ===")
    print(result["answer"])

    print("\n=== 引用来源 ===")

    for sid in result["citations"]:
        item = result["evidence"][sid]
        metadata = item["metadata"]

        print(
            f"[{sid}] {metadata['title']} "
            f"| PDF第{metadata['pdf_page']}页 "
            f"| {item['kind']}"
        )
        print(metadata["source_url"])
        print("原文预览：", item["text"][:350])
        print()

    print("状态：", result["status"])
    print("工具调用次数：", result["tool_calls"])
    print("模型请求轮数：", result["model_rounds"])
    print("总耗时：", result["elapsed_seconds"], "秒")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("运行记录已保存：", path)


if __name__ == "__main__":
    main()
