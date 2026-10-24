
import csv
import re
import logging
from pathlib import Path
from collections import Counter

import pymupdf


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def clean_text(text: str) -> str:
    """保守清洗，保留技术术语、数字与段落。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[^\S\n]+", " ", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_sources(csv_path="data/sources.csv") -> list[dict]:
    """读取 PDF 来源清单。"""
    path = PROJECT_ROOT / csv_path

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "doc_id", "title", "source_url",
            "local_file", "document_version"
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("sources.csv 缺少必要字段")

        sources = list(reader)

    if not sources:
        raise ValueError("sources.csv 为空")

    return sources


def get_blocks(page):
    """
    获取文本块和坐标。
    返回 (x0, y0, x1, y1, text)。
    """
    result = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[:5]
        text = clean_text(text)

        if text:
            result.append((x0, y0, x1, y1, text))

    return result


def margin_key(text):
    """规范化页边缘文本，以识别重复页眉页脚。"""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_margin(block, page_height):
    """判断文本块是否处于页面上下边缘。"""
    _, y0, _, y1, _ = block

    return (
        y1 < page_height * 0.09
        or y0 > page_height * 0.91
    )


def detect_repeated_margins(doc):
    """
    统计同一 PDF 的重复页边缘文本。
    至少出现在 3 个不同页面才判定为候选。
    """
    counts = Counter()

    for page in doc:
        unique = set()

        for block in get_blocks(page):
            text = block[4]

            if (
                is_margin(block, page.rect.height)
                and len(text) <= 120
            ):
                unique.add(margin_key(text))

        counts.update(unique)

    return {
        text for text, count in counts.items()
        if count >= 3
    }


def remove_repeated_margins(blocks, page_height, repeated):
    """只删除已确认在多页重复的页边缘文本块。"""
    cleaned = []

    for block in blocks:
        text = block[4]

        if (
            is_margin(block, page_height)
            and margin_key(text) in repeated
        ):
            continue

        cleaned.append(block)

    return cleaned


def has_two_columns(blocks, page_width):
    """
    判断是否存在明显双栏正文。

    只使用较长的正文文本块进行判断，
    避免把图例和表格碎片误判为双栏。
    """
    mid = page_width / 2

    left = []
    right = []

    for block in blocks:
        x0, y0, x1, y1, text = block

        if len(text) < 80:
            continue

        if x1 < mid - page_width * 0.025:
            left.append(block)

        elif x0 > mid + page_width * 0.025:
            right.append(block)

    if len(left) < 2 or len(right) < 2:
        return False

    # 两栏正文应当在垂直方向有明显重叠
    left_top = min(b[1] for b in left)
    left_bottom = max(b[3] for b in left)

    right_top = min(b[1] for b in right)
    right_bottom = max(b[3] for b in right)

    overlap_height = max(
        0,
        min(left_bottom, right_bottom)
        - max(left_top, right_top)
    )

    shorter_height = min(
        left_bottom - left_top,
        right_bottom - right_top
    )

    return (
        shorter_height > 0
        and overlap_height / shorter_height >= 0.35
    )


def order_blocks(blocks, page_width):
    """
    对明显双栏页采用左栏优先的阅读顺序；
    对普通页面按坐标排序。

    跨越页面中线的宽文本块可作为纵向分区边界。
    """
    if not has_two_columns(blocks, page_width):
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    mid = page_width / 2

    # 宽文本块作为分区边界，如整行标题
    spanning = sorted(
        [
            b for b in blocks
            if b[0] < mid and b[2] > mid
        ],
        key=lambda b: b[1]
    )

    narrow = [
        b for b in blocks
        if b not in spanning
    ]

    # 只使用不与其他跨栏块重叠的分区边界
    ordered = []
    remaining = narrow
    previous_bottom = float("-inf")

    for wide in spanning:
        y0, y1 = wide[1], wide[3]

        if y0 < previous_bottom:
            continue

        section = [
            b for b in remaining
            if b[3] <= y0
        ]

        ordered.extend(order_column_section(section, mid))
        remaining = [b for b in remaining if b not in section]

        ordered.append(wide)
        previous_bottom = y1

    ordered.extend(order_column_section(remaining, mid))
    return ordered


def order_column_section(blocks, mid):
    """同一页面区域中，先左栏后右栏。"""
    left = []
    right = []

    for block in blocks:
        center_x = (block[0] + block[2]) / 2

        if center_x < mid:
            left.append(block)
        else:
            right.append(block)

    return (
        sorted(left, key=lambda b: (b[1], b[0]))
        + sorted(right, key=lambda b: (b[1], b[0]))
    )


def is_suspicious_text(text):
    """
    检查疑似图表碎片。
    仅记录警告，不直接删除。
    """
    words = text.split()

    if len(words) < 15:
        return False

    # 统计大量重复的短标签
    short_labels = sum(
        1 for word in words
        if len(word) <= 5
    )

    return short_labels / len(words) > 0.85


def extract_page_text(page, repeated):
    """解析单页文本，返回正文及布局信息。"""

    blocks = get_blocks(page)

    blocks = remove_repeated_margins(
        blocks,
        page.rect.height,
        repeated
    )

    two_columns = has_two_columns(
        blocks,
        page.rect.width
    )

    ordered = order_blocks(blocks, page.rect.width)

    # 使用空行连接文本块，方便后续按段落分块
    text = "\n\n".join(b[4] for b in ordered)

    text = clean_text(text)

    return text, two_columns


def load_pdf(path: str, metadata: dict) -> list[dict]:
    """逐页解析 PDF，保留来源元数据。"""

    pdf_path = Path(path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as doc:
        repeated = detect_repeated_margins(doc)

        for page_index, page in enumerate(doc):
            text, two_columns = extract_page_text(
                page, repeated
            )

            page_number = page_index + 1

            if not text:
                logger.warning(
                    "空白或图片页: %s 第 %d 页",
                    pdf_path.name, page_number
                )

            if two_columns:
                logger.info(
                    "双栏布局: %s 第 %d 页",
                    pdf_path.name, page_number
                )

            if is_suspicious_text(text):
                logger.warning(
                    "疑似图表碎片: %s 第 %d 页",
                    pdf_path.name, page_number
                )

            pages.append({
                "doc_id": metadata["doc_id"],
                "title": metadata["title"],
                "source_url": metadata["source_url"],
                "document_version": metadata["document_version"],
                "pdf_page": page_number,
                "text": text,
            })

    return pages


def load_all_pdfs() -> list[dict]:
    """按照 sources.csv 加载全部 PDF。"""
    sources = load_sources()
    all_pages = []

    for source in sources:
        pdf_path = PROJECT_ROOT / source["local_file"]

        logger.info("解析文档: %s", source["doc_id"])

        pages = load_pdf(str(pdf_path), source)
        all_pages.extend(pages)

        logger.info(
            "完成 %s，共 %d 页",
            source["doc_id"], len(pages)
        )

    return all_pages
