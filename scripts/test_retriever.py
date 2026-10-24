
from src.retriever import HybridRetriever, rrf_fuse


def show_results(name, results):
    print(f"\n========== {name} ==========")

    for rank, item in enumerate(results, start=1):
        meta = item["metadata"]

        print(f"\nRank: {rank}")
        print(f"Score: {item['score']:.6f}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Document: {meta['title']}")
        print(f"PDF Page: {meta['pdf_page']}")
        print(f"Text: {item['text'][:250]}")


def main():
    print("正在初始化检索器...")
    retriever = HybridRetriever()

    # question = "Wi-Fi 6 中 OFDMA 和 MU-MIMO 有什么区别？"
    question = "26-tone RU 在 Wi-Fi 6 中有什么作用？"

    print("\n问题:", question)

    dense = retriever.dense_search(question, top_k=5)
    bm25 = retriever.bm25_search(question, top_k=5)
    hybrid = retriever.hybrid_search(question, top_k=5)

    show_results("Dense Retrieval", dense)
    show_results("BM25 Retrieval", bm25)
    show_results("Hybrid Retrieval", hybrid)

    # 检查返回结果
    assert dense, "Dense 没有返回结果"
    assert hybrid, "Hybrid 没有返回结果"

    for name, results in [
        ("dense", dense),
        ("bm25", bm25),
        ("hybrid", hybrid),
    ]:
        assert len(results) <= 5

        ids = [item["chunk_id"] for item in results]
        assert len(ids) == len(set(ids)), f"{name} 存在重复 Chunk"

        for item in results:
            assert item["text"]
            assert item["metadata"]["doc_id"]
            assert item["metadata"]["pdf_page"] >= 1

    # 测试 RRF 排名是否符合预期
    a = {"chunk_id": "A", "score": 0.9}
    b = {"chunk_id": "B", "score": 0.8}
    c = {"chunk_id": "C", "score": 5.0}

    fused = rrf_fuse([a, b], [b, c])

    assert fused[0]["chunk_id"] == "B"
    assert len(fused) == 3

    print("\n基础检查通过！")


if __name__ == "__main__":
    main()
