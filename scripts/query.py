
"""
query.py — RAG 命令行问答入口

功能：
    从命令行接收用户问题和检索模式，
    调用 RAGPipeline 完成检索与答案生成。

参数：
    --question：用户问题
    --mode：dense、bm25 或 hybrid
    --retrieve-only：只检索，不调用 LLM
    --top-k：返回的证据数量
"""

import argparse

from src.pipeline import RAGPipeline


def main():

    parser = argparse.ArgumentParser(
        description="Wireless-RAG 命令行问答"
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--retrieve-only",
        action="store_true"
    )

    args = parser.parse_args()

    # 初始化 RAG Pipeline
    pipeline = RAGPipeline()

    # 执行查询
    result = pipeline.query(
        question=args.question,
        mode=args.mode,
        top_k=args.top_k,
        retrieve_only=args.retrieve_only
    )

    if args.retrieve_only:

        print("\n=== 检索结果 ===")

        for i, chunk in enumerate(
            result["contexts"], start=1
        ):
            print(f"\n[S{i}]")
            print("Chunk ID:", chunk["chunk_id"])
            print("Score:", chunk["score"])
            print("Metadata:", chunk["metadata"])
            print("Text:", chunk["text"][:500])

    else:

        print("\n=== 生成答案 ===")
        print(result["answer"])

        print("\n=== 真实来源 ===")

        for source in result["sources"]:

            metadata = source["metadata"]

            print(f"\n[{source['source_id']}]")
            print("Chunk ID:", source["chunk_id"])
            print("Title:", metadata.get("title"))
            print("PDF Page:", metadata.get("pdf_page"))
            print("URL:", metadata.get("source_url"))
            print("Text:", source["text"])

        print("\n=== 无效引用 ===")
        print(result["invalid_citations"])


if __name__ == "__main__":
    main()
