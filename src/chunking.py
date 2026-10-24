
from collections import defaultdict
import re
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 人工检查后确认不适合直接文本分块的页面
EXCLUDED_PAGES = {
    ("spirent_ofdma", 16),
}

def split_long_text(text, tokenizer, max_tokens, overlap):
    """
    对超长段落使用滑动窗口分块。
    确保最终文本重新编码后不超过 max_tokens。
    """

    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap 必须满足 0 <= overlap < max_tokens")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    start = 0

    while start < len(token_ids):

        end = min(start + max_tokens, len(token_ids))

        # 解码之后重新编码，检查真实长度
        while end > start:

            window = token_ids[start:end]

            chunk_text = tokenizer.decode(
                window,
                skip_special_tokens=True
            ).strip()

            actual_tokens = len(tokenizer.encode(
                chunk_text,
                add_special_tokens=False
            ))

            if chunk_text and actual_tokens <= max_tokens:
                break

            end -= 1

        if end <= start:
            raise ValueError(
                f"无法在 {max_tokens} tokens 内完成分块"
            )

        yield chunk_text

        if end >= len(token_ids):
            break

        # 下一块从上一块结尾向前 overlap 个 token
        next_start = max(start + 1, end - overlap)

        start = next_start

def is_low_quality_chunk(text: str) -> bool:
    """
    识别明显低质量的 Chunk。
    使用保守规则，避免删除正常技术正文。
    """

    words = text.split()

    if not words:
        return True

    # 1. 极短内容：通常是封面标题、页码等
    if len(words) < 8:
        return True

    # 2. 识别表格中常见的孤立标签与数字
    table_tokens = 0

    for word in words:
        token = word.strip(".,:;()[]")

        if (
            re.fullmatch(r"\d+(\.\d+)?", token)
            or re.fullmatch(r"RU\d+", token, re.I)
            or re.fullmatch(r"AID\d+", token, re.I)
            or token.upper() in {"NS"}
        ):
            table_tokens += 1

    # 3. 表格标签占比过高，判定为碎片
    table_ratio = table_tokens / len(words)

    if len(words) >= 15 and table_ratio > 0.65:
        return True

    return False

def split_pages(pages, tokenizer):
    """
    按页分块，保留原始文档元数据。
    """
    chunks = []

    for page in pages:
        # 暂时排除复杂表格页
        page_key = (
            page["doc_id"],
            page["pdf_page"]
        )

        if page_key in EXCLUDED_PAGES:
            continue

        text = page["text"].strip()

        if not text:
            continue

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        page_chunks = []
        current = ""

        def flush():
            nonlocal current

            if current:
                page_chunks.append(current)
                current = ""

        for paragraph in paragraphs:
            paragraph_tokens = tokenizer.encode(
                paragraph,
                add_special_tokens=False
            )

            # 情况 1：单个段落超过上限
            if len(paragraph_tokens) > CHUNK_SIZE:
                flush()

                for part in split_long_text(
                    paragraph,
                    tokenizer,
                    CHUNK_SIZE,
                    CHUNK_OVERLAP
                ):
                    page_chunks.append(part)

                continue

            # 情况 2：尝试加入当前 Chunk
            candidate = (
                current + "\n\n" + paragraph
                if current else paragraph
            )

            token_count = len(tokenizer.encode(
                candidate,
                add_special_tokens=False
            ))

            if token_count <= CHUNK_SIZE:
                current = candidate
            else:
                flush()
                current = paragraph

        flush()

        valid_chunks = [
            text for text in page_chunks
            if not is_low_quality_chunk(text)
        ]

        # 生成稳定的 chunk_id
        for index, chunk_text in enumerate(valid_chunks, 1):
            chunk_id = (
                f"{page['doc_id']}_"
                f"p{page['pdf_page']:03d}_"
                f"c{index:03d}"
            )  # e.g. cisco_wifi6_p005_c003

            chunk = {
                "chunk_id": chunk_id,
                "doc_id": page["doc_id"],
                "title": page["title"],
                "source_url": page["source_url"],
                "document_version": page["document_version"],
                "pdf_page": page["pdf_page"],
                "text": chunk_text,
            }

            chunks.append(chunk)

    return chunks
