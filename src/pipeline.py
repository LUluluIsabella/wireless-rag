
"""
pipeline.py — RAG 流程控制模块

功能：
    将 Retriever 和 Generator 串联，
    实现从用户问题到最终答案的完整 RAG 流程。

主要函数：
    1. __init__()
       初始化并复用 HybridRetriever。

    2. query()
       根据指定检索模式获取 Top-K Chunks，
       支持仅检索或调用 LLM 生成带引用的答案。

流程：
    Question → Retrieval → Context → Generation → Answer
"""

from src.retriever import HybridRetriever
from src.generator import generate_answer


class RAGPipeline:

    def __init__(self):
        # 初始化一次检索器
        # 加载 FAISS、Embedding 模型及 BM25
        self.retriever = HybridRetriever()

        # 只要同一个 RAGPipeline 实例被复用，检索模型就不需要每次提问时重新加载。
        # 这在真实 Agent 项目中也非常重要，因为反复加载模型会严重影响响应时间。

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        retrieve_only: bool = False
    ) -> dict:

        # 检查检索模式是否合法
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"不支持的检索模式: {mode}")

        if not question.strip():
            raise ValueError("问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        # 根据指定模式执行检索
        if mode == "dense":
            contexts = self.retriever.dense_search(
                question, top_k=top_k
            )

        elif mode == "bm25":
            contexts = self.retriever.bm25_search(
                question, top_k=top_k
            )

        else:
            contexts = self.retriever.hybrid_search(
                question, top_k=top_k
            )

        # 仅检索模式：不调用 LLM
        if retrieve_only:
            return {
                "question": question,
                "mode": mode,
                "contexts": contexts
            }

        # 将检索结果交给 Generator
        result = generate_answer(
            question=question,
            contexts=contexts
        )

        # 返回最终 RAG 结果
        return {
            "question": question,
            "mode": mode,
            "answer": result["answer"],
            "citations": result["citations"],
            "sources": result["sources"],
            "invalid_citations": result["invalid_citations"]
        }
