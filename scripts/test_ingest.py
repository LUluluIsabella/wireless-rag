
from collections import Counter

from src.ingest import load_all_pdfs

# 基础的pdf解析测试脚本，检查 PDF 解析是否正常工作。
def main():

    # 解析全部 PDF
    pages = load_all_pdfs()

    print("\n========== PDF 解析统计 ==========")

    print("总页数:", len(pages))

    # 统计每份文档的页数
    counts = Counter(page["doc_id"] for page in pages)

    for doc_id, count in counts.items():
        print(f"{doc_id}: {count} pages")

    # 检查页数
    assert counts["cisco_wifi6"] == 16
    assert counts["zte_wifi6"] == 35
    assert counts["spirent_ofdma"] == 20

    assert len(pages) == 71

    print("\n========== 文本抽查 ==========")

    # 选择每份 PDF 的前几页
    for doc_id in counts:

        doc_pages = [
            p for p in pages
            if p["doc_id"] == doc_id
        ]

        for page in doc_pages[:4]:

            print("\n------------------------------")
            print("文档:", page["doc_id"])
            print("PDF 页码:", page["pdf_page"])
            print("字符数:", len(page["text"]))
            print("内容预览:")
            print(page["text"][:300])

    print("\n基础检查通过！")


if __name__ == "__main__":
    main()
