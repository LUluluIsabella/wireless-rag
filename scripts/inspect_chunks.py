"""
inspect_chunks.py — 查看 RAG 文档片段

功能：
    读取 storage/chunks.jsonl，
    展示每个 Chunk 的 ID、文档标题、页码和文本。

用途：
    辅助人工构建 RAG Evaluation 数据集，
    确保 gold_chunk_ids 来自真实文档。
"""

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = ROOT / "storage" / "chunks.jsonl"


def load_chunks():
    """读取所有 Chunk。"""

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="根据关键词过滤 Chunk"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="最多显示多少个 Chunk"
    )

    args = parser.parse_args()

    chunks = load_chunks()

    print(f"总 Chunk 数量：{len(chunks)}")

    count = 0

    for chunk in chunks:

        text = chunk.get("text", "")

        # 如果指定关键词，只显示包含关键词的 Chunk
        if args.keyword:
            if args.keyword.lower() not in text.lower():
                continue

        print("\n" + "=" * 60)

        print("Chunk ID:", chunk.get("chunk_id"))

        print("Document:", chunk.get("title"))

        print("PDF Page:", chunk.get("pdf_page"))

        print("\nText:")
        print(text)

        count += 1

        if count >= args.limit:
            break

    print(f"\n本次显示：{count} 个 Chunk")


if __name__ == "__main__":
    main()