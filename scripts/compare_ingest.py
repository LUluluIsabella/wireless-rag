
import pymupdf

from src.ingest import (
    load_sources,
    load_pdf,
    PROJECT_ROOT,
)


def main():
    source = next(
        s for s in load_sources()
        if s["doc_id"] == "spirent_ofdma"
    )

    path = PROJECT_ROOT / source["local_file"]

    # 优化后的解析
    new_pages = load_pdf(str(path), source)

    # 对比第 2、8、10、16、18 页
    inspect_pages = [2, 8, 10, 16, 18]

    with pymupdf.open(path) as doc:
        for page_num in inspect_pages:
            old_text = doc[page_num - 1].get_text(
                "text", sort=True
            )

            new_text = new_pages[page_num - 1]["text"]

            print("\n" + "=" * 65)
            print(f"Spirent PDF 第 {page_num} 页")

            print("\n[原始提取]")
            print(old_text[:800])

            print("\n[优化后提取]")
            print(new_text[:800])

            print("\n字符数量:")
            print("原始:", len(old_text))
            print("优化:", len(new_text))


if __name__ == "__main__":
    main()
