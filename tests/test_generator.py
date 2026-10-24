
"""
Generator 单元测试

测试：
1. 普通引用提取
2. 区间引用提取
3. 引用去重
4. 无效引用检测
5. 无效引用答案拦截
"""

from unittest.mock import MagicMock, patch

from src.generator import (
    extract_citations,
    generate_answer,
)


# =========================
# 1. 引用解析测试
# =========================

def test_single_citation():
    answer = "OFDMA 支持多用户传输 [S1]"

    result = extract_citations(answer)

    assert result == ["S1"]


def test_multiple_citations():
    answer = "OFDMA [S1][S2]，MU-MIMO [S3]"

    result = extract_citations(answer)

    assert result == ["S1", "S2", "S3"]


def test_range_citations():
    answer = "相关证据为 [S1]–[S5]"

    result = extract_citations(answer)

    assert result == ["S1", "S2", "S3", "S4", "S5"]


def test_compact_range():
    answer = "相关证据为 [S1-S5]"

    result = extract_citations(answer)

    assert result == ["S1", "S2", "S3", "S4", "S5"]


def test_duplicate_citations():
    answer = "证据 [S1][S2]，再次引用 [S1]"

    result = extract_citations(answer)

    assert result == ["S1", "S2"]


def test_invalid_citation_extraction():
    answer = "错误引用 [S99]"

    result = extract_citations(answer)

    assert result == ["S99"]


# =========================
# 2. 无效引用防御测试
# =========================

def test_invalid_citation_rejected():

    # 模拟检索器返回两条证据
    contexts = [
        {
            "chunk_id": "test_chunk_001",
            "text": "OFDMA divides a channel into RUs.",
            "metadata": {
                "doc_id": "test_wifi6",
                "pdf_page": 1,
            },
        },
        {
            "chunk_id": "test_chunk_002",
            "text": "MU-MIMO uses spatial multiplexing.",
            "metadata": {
                "doc_id": "test_wifi6",
                "pdf_page": 2,
            },
        },
    ]

    # 模拟 LLM 返回包含非法引用的答案
    mock_response = MagicMock()

    mock_response.choices[0].message.content = (
        "OFDMA 使用 RU [S1]，"
        "另一个结论来自不存在的来源 [S99]。"
    )

    with patch.dict(
        "os.environ",
        {
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "LLM_BASE_URL": "https://example.com/v1",
        },
    ):
        with patch("src.generator.OpenAI") as mock_client:

            mock_client.return_value.chat.completions.create.return_value = (
                mock_response
            )

            result = generate_answer(
                question="OFDMA 是什么？",
                contexts=contexts,
            )

    # S1 存在，因此是合法引用
    assert result["citations"] == ["S1"]

    # S99 不存在，因此必须被识别
    assert result["invalid_citations"] == ["S99"]

    # 不允许原始错误答案直接返回给用户
    assert "本次回答未通过引用校验" in result["answer"]

    # 不允许错误答案携带有效来源映射
    assert result["sources"] == []
