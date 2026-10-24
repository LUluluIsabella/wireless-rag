
import json
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

from src.ingest import load_all_pdfs
from src.chunking import split_pages, CHUNK_SIZE, is_low_quality_chunk


MODEL_NAME = "intfloat/multilingual-e5-small"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():

    print("========== Step 1: 加载 Tokenizer ==========")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizer 加载成功！")

    print("\n========== Step 2: 加载 PDF 文本 ==========")

    pages = load_all_pdfs()

    print(f"PDF 页面总数: {len(pages)}")

    print("\n========== Step 3: 文档分块 ==========")

    chunks = split_pages(pages, tokenizer)

    print(f"Chunk 总数: {len(chunks)}")

    # ===== 新增：低质量 Chunk 检查 =====

    low_quality_chunks = [
        chunk for chunk in chunks
        if is_low_quality_chunk(chunk["text"])
    ]

    print("仍存在疑似低质量 Chunk:", len(low_quality_chunks))

    # 如果仍有低质量 Chunk，打印前 5 个供检查
    for chunk in low_quality_chunks[:5]:
        print("\n疑似低质量 Chunk:")
        print("Chunk ID:", chunk["chunk_id"])
        print("文本预览:", chunk["text"][:200])


    counts = Counter(chunk["doc_id"] for chunk in chunks)

    for doc_id, count in counts.items():
        print(f"{doc_id}: {count} chunks")

    print("\n========== Step 4: 检查 Token 长度 ==========")

    assert chunks, "没有生成任何 Chunk"

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk ID 存在重复"

    lengths = []

    for chunk in chunks:
        token_count = len(tokenizer.encode(
            chunk["text"],
            add_special_tokens=False
        ))

        lengths.append(token_count)
        if token_count > CHUNK_SIZE:
            print("\n发现超限 Chunk:")
            print("Chunk ID:", chunk["chunk_id"])
            print("实际 Token 数:", token_count)
            print("允许 Token 数:", CHUNK_SIZE)
            print("文本预览:", repr(chunk["text"][:300]))

        assert token_count <= CHUNK_SIZE, (
            f"{chunk['chunk_id']} 超过长度限制: {token_count}"
        )

        # 检查后续 E5 Embedding 的实际输入长度
        model_input = "passage: " + chunk["text"]

        model_token_count = len(tokenizer.encode(
            model_input,
            add_special_tokens=True
        ))

        assert model_token_count <= 512, (
            f"{chunk['chunk_id']} 超过 E5 模型上限: "
            f"{model_token_count}"
        )
        assert chunk["text"].strip(), "发现空 Chunk"

    print("最大 Token 数:", max(lengths))
    print("平均 Token 数:", round(sum(lengths) / len(lengths), 2))
    print("长度检查通过！")

    print("\n========== Step 5: 保存 JSONL ==========")

    output_path = PROJECT_ROOT / "data/processed/chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"分块结果已保存: {output_path}")

    print("\n========== Step 6: 输出样例 ==========")

    for chunk in chunks[:3]:
        print("-" * 50)
        print("Chunk ID:", chunk["chunk_id"])
        print("PDF Page:", chunk["pdf_page"])
        print("Text:", chunk["text"][:300])

    print("\n基础检查通过！")


if __name__ == "__main__":
    main()
