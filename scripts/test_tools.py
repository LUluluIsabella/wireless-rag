# 用途：独立验证两个真实工具，不调用LLM。
# 检查参数拒绝、真实搜索、来源映射和页面读取。
# 使用搜索结果中的实际doc_id和页码，不手写猜测来源。
# 本脚本属于本地集成验收，需要已有索引、模型和原始PDF。

import json

from src.tools import WifiTools


def check_error(result: dict, expected_code: str) -> None:
    assert result["ok"] is False, result
    assert result["error"]["code"] == expected_code, result


def main() -> None:
    tools = WifiTools()

    print("=== 1. 参数与文档校验 ===")

    check_error(
        tools.search_wifi_docs(""),
        "INVALID_ARGUMENT",
    )
    check_error(
        tools.search_wifi_docs("OFDMA", top_k=100),
        "INVALID_ARGUMENT",
    )
    check_error(
        tools.search_wifi_docs("OFDMA", top_k=True),
        "INVALID_ARGUMENT",
    )
    check_error(
        tools.get_pdf_page("cisco_wifi6", 0),
        "INVALID_ARGUMENT",
    )
    check_error(
        tools.get_pdf_page("__missing_document__", 1),
        "DOC_NOT_FOUND",
    )

    print("参数与文档校验通过")

    print("\n=== 2. 真实Hybrid检索 ===")

    result = tools.search_wifi_docs(
        "OFDMA 和 MU-MIMO 有什么区别？",
        top_k=3,
    )

    assert result["ok"], result

    chunks = result["data"]["results"]
    assert 1 <= len(chunks) <= 3, result

    for item in chunks:
        assert item["chunk_id"]
        assert item["text"].strip()
        assert item["metadata"]["doc_id"]
        assert item["metadata"]["pdf_page"] >= 1

        print(
            item["chunk_id"],
            item["metadata"]["doc_id"],
            item["metadata"]["pdf_page"],
        )

    print("\n=== 3. 读取首条结果对应页面 ===")

    metadata = chunks[0]["metadata"]

    page_result = tools.get_pdf_page(
        metadata["doc_id"],
        metadata["pdf_page"],
    )

    assert page_result["ok"], page_result

    page = page_result["data"]

    assert page["text"].strip()
    assert page["metadata"]["doc_id"] == metadata["doc_id"]
    assert page["metadata"]["pdf_page"] == metadata["pdf_page"]

    print(json.dumps(page["metadata"], ensure_ascii=False, indent=2))
    print("正文预览：")
    print(page["text"][:600])
    print("是否截断：", page["truncated"])

    print("\n=== 4. 超范围页码校验 ===")

    check_error(
        tools.get_pdf_page(
            metadata["doc_id"],
            page["metadata"]["page_count"] + 1,
        ),
        "PAGE_OUT_OF_RANGE",
    )

    print("\n全部工具集成检查通过")


if __name__ == "__main__":
    main()
