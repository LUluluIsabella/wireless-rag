
"""
evaluate.py — RAG 检索性能评测模块

功能：
    使用人工标注的 Gold Chunks，
    评估 Dense、BM25、Hybrid 三种检索模式的效果。

主要函数：
    1. load_dataset()
       读取 JSONL 评测集。

    2. calculate_metrics()
       计算单个问题的 Hit@K 和 RR@K。

    3. run_retrieval()
       根据指定模式调用 Retriever。

    4. evaluate()
       执行完整评测，统计 Hit@K、MRR@K 和延迟。

    5. main()
       解析命令行参数并保存实验结果。

评测指标：
    Hit@5：Top 5 是否命中至少一个 Gold Chunk。
    MRR@5：正确证据首次出现排名的平均倒数。
    Latency：单次检索耗时，不包含 LLM 生成时间。
"""

import argparse
import json
import time

from pathlib import Path
from statistics import mean

from src.retriever import HybridRetriever


ROOT = Path(__file__).resolve().parent.parent


def load_dataset(dataset_path: Path) -> list[dict]:
    """
    读取 JSONL 评测数据集。
    支持 UTF-8 BOM，并定位 JSON 格式错误。
    """

    dataset = []

    # utf-8-sig 可以自动处理 UTF-8 BOM
    with dataset_path.open(
        "r", encoding="utf-8-sig"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            # 去除首尾空白字符
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"\nJSONL 格式错误！"
                    f"\n文件：{dataset_path}"
                    f"\n文件行号：{line_number}"
                    f"\n错误原因：{e}"
                    f"\n错误内容：{line[:200]}"
                ) from e

            dataset.append(item)

    return dataset


def calculate_metrics(
    retrieved_ids: list[str],
    gold_chunk_ids: list[str],
    k: int = 5
) -> dict:
    """
    计算单个问题的 Hit@K 和 RR@K。

    只适用于可回答且具有 Gold Chunks 的问题。
    """

    gold_set = set(gold_chunk_ids)

    # 防止将不存在 Gold 证据的问题误算为失败
    if not gold_set:
        raise ValueError("可回答问题必须提供 Gold Chunk IDs")

    # 只检查前 K 个结果
    top_k_ids = retrieved_ids[:k]

    # 默认未命中
    hit = 0
    reciprocal_rank = 0.0
    first_hit_rank = None

    for rank, chunk_id in enumerate(top_k_ids, start=1):

        if chunk_id in gold_set:

            hit = 1
            reciprocal_rank = 1.0 / rank
            first_hit_rank = rank

            # 只考虑第一个正确证据的排名
            break

    return {
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "first_hit_rank": first_hit_rank
    }


def run_retrieval(
    retriever,
    question: str,
    mode: str,
    top_k: int
) -> list[dict]:
    """根据检索模式调用对应的检索方法。"""

    if mode == "dense":
        return retriever.dense_search(
            question, top_k=top_k
        )

    elif mode == "bm25":
        return retriever.bm25_search(
            question, top_k=top_k
        )

    elif mode == "hybrid":
        return retriever.hybrid_search(
            question, top_k=top_k
        )

    else:
        raise ValueError(f"不支持的检索模式: {mode}")


def evaluate(
    retriever,
    dataset: list[dict],
    mode: str,
    top_k: int = 5
) -> dict:
    """
    对整个评测集运行检索实验。

    不可回答问题单独记录，
    不纳入 Hit@K 和 MRR@K 的分母。
    """

    results = []

    hits = []
    reciprocal_ranks = []
    latencies = []

    for item in dataset:

        question = item["question"]
        answerable = item["answerable"]
        gold_ids = item.get("gold_chunk_ids", [])

        # 不可回答问题不参与检索命中率评测
        if not answerable:
            results.append({
                "id": item["id"],
                "question": question,
                "answerable": False,
                "status": "skipped_unanswerable"
            })
            continue

        if not gold_ids:
            raise ValueError(
                f"{item['id']} 可回答但缺少 Gold Chunks"
            )

        # 使用 perf_counter 测量检索耗时
        start_time = time.perf_counter()

        retrieved = run_retrieval(
            retriever=retriever,
            question=question,
            mode=mode,
            top_k=top_k
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in retrieved
        ]

        # 计算当前问题的指标
        metrics = calculate_metrics(
            retrieved_ids=retrieved_ids,
            gold_chunk_ids=gold_ids,
            k=top_k
        )

        hits.append(metrics["hit"])
        reciprocal_ranks.append(
            metrics["reciprocal_rank"]
        )
        latencies.append(latency_ms)

        # 保留每道题的完整检索排名，方便错误分析
        results.append({
            "id": item["id"],
            "question": question,
            "category": item.get("category"),
            "answerable": True,
            "gold_chunk_ids": gold_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "hit": metrics["hit"],
            "reciprocal_rank": metrics["reciprocal_rank"],
            "first_hit_rank": metrics["first_hit_rank"],
            "latency_ms": latency_ms
        })

        print(
            f"{item['id']} | "
            f"Hit={metrics['hit']} | "
            f"RR={metrics['reciprocal_rank']:.3f} | "
            f"Latency={latency_ms:.2f}ms"
        )

    summary = {
        "mode": mode,
        "top_k": top_k,
        "total_questions": len(dataset),
        "evaluated_questions": len(hits),
        "skipped_unanswerable": (
            len(dataset) - len(hits)
        ),
        "hit_at_k": mean(hits) if hits else None,
        "mrr_at_k": (
            mean(reciprocal_ranks)
            if reciprocal_ranks else None
        ),
        "avg_latency_ms": (
            mean(latencies)
            if latencies else None
        ),
        "results": results
    }

    return summary


def main():

    parser = argparse.ArgumentParser(
        description="Wireless-RAG Retrieval Evaluation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        required=True
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")

    dataset_path = ROOT / args.dataset

    dataset = load_dataset(dataset_path)

    # 检索器只初始化一次
    retriever = HybridRetriever()

    # Warm-up：预热检索器，排除首次初始化开销
    # 仅用于性能计时，不参与 Hit@5 / MRR@5 计算

    warmup_question = "Wi-Fi 6 OFDMA"

    # 预热 E5 Embedding 和 FAISS 检索
    retriever.dense_search(
        warmup_question,
        top_k=5
    )

    # 预热 Jieba 分词和 BM25 查询
    retriever.bm25_search(
        warmup_question,
        top_k=5
    )

    # 预热完整的 Dense + BM25 + RRF 流程
    retriever.hybrid_search(
        warmup_question,
        top_k=5
    )

    summary = evaluate(
        retriever=retriever,
        dataset=dataset,
        mode=args.mode,
        top_k=args.top_k
    )

    print("\n========== Evaluation Summary ==========")

    print("Mode:", summary["mode"])
    print("Evaluated:", summary["evaluated_questions"])
    print("Skipped:", summary["skipped_unanswerable"])

    hit = summary["hit_at_k"]
    mrr = summary["mrr_at_k"]
    latency = summary["avg_latency_ms"]

    if hit is not None:
        print(f"Hit@{args.top_k}: {hit:.4f}")
        print(f"MRR@{args.top_k}: {mrr:.4f}")
        print(f"Avg Latency: {latency:.2f} ms")
    else:
        print("没有可回答问题，无法计算检索指标。")

    # 保存详细评测结果
    if args.output:

        output_path = ROOT / args.output

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
