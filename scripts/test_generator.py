
from src.generator import generate_answer


def main():

    # 模拟 Retriever 返回的两个 Chunks
    # 注意：以下文本只是测试样例
    contexts = [
        {
            "chunk_id": "test_chunk_001",
            "score": 0.95,
            "text": (
                "OFDMA divides a wireless channel into "
                "smaller resource units, allowing multiple "
                "users to transmit simultaneously."
            ),
            "metadata": {
                "doc_id": "test_wifi6",
                "pdf_page": 5,
                "title": "Test Wi-Fi 6 Document"
            }
        },
        {
            "chunk_id": "test_chunk_002",
            "score": 0.90,
            "text": (
                "MU-MIMO uses multiple antennas to "
                "communicate with multiple users "
                "through spatial multiplexing."
            ),
            "metadata": {
                "doc_id": "test_wifi6",
                "pdf_page": 6,
                "title": "Test Wi-Fi 6 Document"
            }
        }
    ]

    question = "OFDMA 和 MU-MIMO 有什么区别？"

    result = generate_answer(
        question=question,
        contexts=contexts
    )

    print("\n=== 生成答案 ===")
    print(result["answer"])

    print("\n=== 引用编号 ===")
    print(result["citations"])

    print("\n=== 真实来源 ===")
    for source in result["sources"]:
        print(source["source_id"])
        print(source["chunk_id"])
        print(source["metadata"])

    print("\n=== 无效引用 ===")
    print(result["invalid_citations"])


if __name__ == "__main__":
    main()
