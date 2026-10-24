
"""
Generation Evaluation

评测流程：
1. 读取生成评测数据集
2. 使用 Hybrid Retriever 检索 Top 5 证据
3. 调用 Generator 生成答案
4. 保存答案、引用及完整检索证据
5. 自动检查引用有效性
6. 保留人工评测字段
"""

import argparse
import json
from pathlib import Path

from src.pipeline import RAGPipeline
from src.generator import generate_answer


def load_dataset(path: str) -> list[dict]:
    """读取 JSONL 格式的评测数据集。"""

    dataset = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    return dataset


def evaluate_generation(
    dataset_path: str,
    output_path: str,
    mode: str = "hybrid",
    top_k: int = 5,
):
    """执行完整的生成评测。"""

    dataset = load_dataset(dataset_path)

    # 初始化一次 Pipeline，避免重复加载检索模型
    pipeline = RAGPipeline()

    results = []

    for item in dataset:

        question_id = item["id"]
        question = item["question"]

        print(f"\n正在评测：{question_id}")
        print(f"问题：{question}")

        try:
            # Step 1：只执行检索 获取真实证据
            retrieval = pipeline.query(
                question=question,
                mode=mode,
                top_k=top_k,
                retrieve_only=True,
            )

            contexts = retrieval["contexts"]

            # Step 2：复用现有 Generator 让 LLM 根据证据回答
            generation = generate_answer(
                question=question,
                contexts=contexts,
            )

            # Step 3：提取生成结果
            answer = generation["answer"]
            citations = generation["citations"]
            sources = generation["sources"]
            invalid_citations = generation["invalid_citations"]

            # 自动判断引用编号是否全部有效
            citation_valid = len(invalid_citations) == 0

            result = {
                "id": question_id,
                "question": question,
                "answerable": item["answerable"],
                "reference_answer": item.get("reference_answer"),
                "gold_chunk_ids": item.get("gold_chunk_ids", []),

                "generated_answer": answer,
                "citations": citations,
                "invalid_citations": invalid_citations,
                "citation_valid": citation_valid,

                # 保存完整 Top 5 证据，便于人工核查
                "retrieved_contexts": contexts,
                "sources": sources,

                # 等待人工审核
                "human_evaluation": {
                    "citation_support": None,
                    "abstention_correct": None,
                    "notes": "",
                },
                "error": None,
            }

            print(f"引用：{citations}")
            print(f"引用有效：{citation_valid}")

        except Exception as e:
            # 单题失败不影响其他题继续评测
            print(f"评测失败：{type(e).__name__}: {e}")

            result = {
                "id": question_id,
                "question": question,
                "answerable": item["answerable"],
                "error": f"{type(e).__name__}: {e}",
            }

        results.append(result)

        # 每题完成后保存检查点，避免 API 中断丢失结果
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n========== Generation Evaluation 完成 ==========")

    success_count = sum(
        1 for result in results
        if result.get("error") is None
    )

    print(f"总题数：{len(results)}")
    print(f"成功执行：{success_count}")
    print(f"执行失败：{len(results) - success_count}")
    print(f"结果已保存：{output_path}")
    print("Citation Support 和 Abstention 仍需人工审核。")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="evaluation/generation_test.jsonl",
    )

    parser.add_argument(
        "--output",
        default="evaluation/generation_results.json",
    )

    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    evaluate_generation(
        dataset_path=args.dataset,
        output_path=args.output,
        mode=args.mode,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
