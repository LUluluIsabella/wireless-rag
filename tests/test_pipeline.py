
"""
RAG Pipeline 单元测试

测试内容：
1. 空问题
2. 非法检索模式
3. 非法 Top-K
4. retrieve_only 不调用 LLM
5. 正常生成结果
"""

from unittest.mock import patch

import pytest

from src.pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    """绕过真实检索器初始化。"""
    with patch("src.pipeline.HybridRetriever"):
        return RAGPipeline()


def test_empty_question(pipeline):
    with pytest.raises(ValueError, match="问题不能为空"):
        pipeline.query("   ")


def test_invalid_mode(pipeline):
    with pytest.raises(ValueError, match="不支持的检索模式"):
        pipeline.query(
            question="OFDMA 是什么？",
            mode="invalid",
        )


def test_invalid_top_k(pipeline):
    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        pipeline.query(
            question="OFDMA 是什么？",
            top_k=0,
        )


def test_retrieve_only(pipeline):

    contexts = [
        {
            "chunk_id": "test_001",
            "text": "OFDMA divides channels into RUs.",
            "metadata": {},
        }
    ]

    pipeline.retriever.hybrid_search.return_value = contexts

    with patch("src.pipeline.generate_answer") as mock_generate:

        result = pipeline.query(
            question="OFDMA 是什么？",
            retrieve_only=True,
        )

        assert result["contexts"] == contexts
        assert result["mode"] == "hybrid"

        # 仅检索时不应该调用 LLM
        mock_generate.assert_not_called()


def test_normal_generation(pipeline):

    contexts = [
        {
            "chunk_id": "test_001",
            "text": "OFDMA divides channels into RUs.",
            "metadata": {},
        }
    ]

    pipeline.retriever.hybrid_search.return_value = contexts

    mock_result = {
        "answer": "OFDMA 将信道划分为 RU [S1]",
        "citations": ["S1"],
        "sources": [
            {
                "source_id": "S1",
                "chunk_id": "test_001",
                "text": contexts[0]["text"],
                "metadata": {},
            }
        ],
        "invalid_citations": [],
    }

    with patch(
        "src.pipeline.generate_answer",
        return_value=mock_result,
    ) as mock_generate:

        result = pipeline.query(
            question="OFDMA 是什么？",
            mode="hybrid",
            top_k=5,
        )

        assert result["answer"] == mock_result["answer"]
        assert result["citations"] == ["S1"]
        assert result["invalid_citations"] == []

        mock_generate.assert_called_once_with(
            question="OFDMA 是什么？",
            contexts=contexts,
        )
