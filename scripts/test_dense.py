# 输入中文问题，返回相关的英文 Wi-Fi 6 技术文档片段。

import json
from pathlib import Path

import faiss

from src.embedding import E5Embedding


ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"


def load_chunks():
    with (STORAGE / "chunks.jsonl").open(
        "r", encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f if line.strip()
        ]


def main():

    print("========== 加载索引 ==========")

    index = faiss.read_index(
        str(STORAGE / "dense.faiss")
    )

    chunks = load_chunks()

    with (STORAGE / "index_config.json").open(
        "r", encoding="utf-8"
    ) as f:
        config = json.load(f)

    assert index.ntotal == len(chunks)
    assert len(chunks) == config["chunk_count"]
    assert index.d == config["embedding_dimension"]

    # 使用建索引时相同的 Embedding 模型
    embedding = E5Embedding(config["embedding_model"])

    question = "Wi-Fi 6 中 OFDMA 和 MU-MIMO 有什么区别？"

    print("\n用户问题:", question)

    print("\n========== Query Embedding ==========")

    query_vector = embedding.encode_query(question)

    print("查询向量 Shape:", query_vector.shape)

    print("\n========== FAISS 检索 ==========")

    top_k = min(5, index.ntotal) # 输出排名前 5 的 Chunk

    scores, indices = index.search(query_vector, top_k)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        print("\n" + "=" * 60)
        print("Rank:", rank)
        print("Score:", round(float(score), 4))
        print("Chunk ID:", chunk["chunk_id"])
        print("Document:", chunk["title"])
        print("PDF Page:", chunk["pdf_page"])
        print("Text:")
        print(chunk["text"][:600])

    print("\nDense 检索测试完成！")


if __name__ == "__main__":
    main()
