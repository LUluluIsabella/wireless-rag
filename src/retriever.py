# 函数	           作用
# tokenize()	对中英文文本进行分词
# expand_query()	将中文 Wi-Fi 术语扩展为英文
# dense_search()	执行 FAISS 向量检索
# bm25_search()	执行关键词检索
# rrf_fuse()	融合两组结果
# hybrid_search()	对外提供混合检索入口


import json
import re
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.embedding import E5Embedding


ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"

DENSE_TOP_K = 20
BM25_TOP_K = 20
CONTEXT_TOP_K = 5
RRF_K = 60


# Wi-Fi 专业术语：用于中文查询英文文档
TERM_MAPPING = {
    "资源单元": ["resource unit", "RU"],
    "目标唤醒时间": ["target wake time", "TWT"],
    "正交频分多址": ["OFDMA"],
    "多用户多输入多输出": ["MU-MIMO"],
    "基本服务集着色": ["BSS Coloring"],
}


# 优先完整识别技术编号、缩写和连字符术语
TOKEN_PATTERN = re.compile(
    r"802\.\d+[a-z]*|"
    r"[a-z0-9]+(?:-[a-z0-9]+)+|"
    r"[a-z]+[0-9]*|"
    r"\d+|"
    r"[\u4e00-\u9fff]+",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """
    中英文混合分词。
    保护 802.11ax、MU-MIMO、26-tone 等术语。
    """
    tokens = []

    for match in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            tokens.extend(
                word.strip()
                for word in jieba.lcut(match)
                if word.strip()
            )
        else:
            tokens.append(match)

    return tokens


def expand_query(question: str) -> str:
    """
    给 BM25 查询增加 Wi-Fi 术语的英文表达。
    不修改原始问题。
    只在 BM25 路径进行查询扩展。Dense 路径保留原始中文问题
    """
    extra = []

    for chinese_term, english_terms in TERM_MAPPING.items():
        if chinese_term in question:
            extra.extend(english_terms)

    return question + " " + " ".join(extra)


def load_chunks():
    path = STORAGE / "chunks.jsonl"

    with path.open("r", encoding="utf-8") as f:
        chunks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not chunks:
        raise ValueError("Chunk 数据为空")

    ids = [c["chunk_id"] for c in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("存在重复 chunk_id")

    return chunks


def format_result(chunk, score):
    """统一 Dense 和 BM25 的输出格式。"""
    return {
        "chunk_id": chunk["chunk_id"],
        "score": float(score),
        "text": chunk["text"],
        "metadata": {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source_url": chunk["source_url"],
            "document_version": chunk["document_version"],
            "pdf_page": chunk["pdf_page"],
        },
    }


def rrf_fuse(dense_results, bm25_results, k=RRF_K):
    """
    Reciprocal Rank Fusion。
    按 chunk_id 融合，重复片段仅保留一次。
    """
    if k <= 0:
        raise ValueError("RRF k 必须大于 0")

    scores = {}
    documents = {}

    for results in (dense_results, bm25_results):
        seen = set()

        for rank, item in enumerate(results, start=1):
            cid = item["chunk_id"]

            # 避免同一路重复片段重复计分
            if cid in seen:
                continue

            seen.add(cid)

            scores[cid] = (
                scores.get(cid, 0.0)
                + 1.0 / (k + rank)
            )

            documents[cid] = item

    ranked_ids = sorted(
        scores,
        key=lambda cid: (-scores[cid], cid),
    )

    return [
        {
            **documents[cid],
            "score": scores[cid],
        }
        for cid in ranked_ids
    ]


class HybridRetriever:
    def __init__(self):
        # 加载与 FAISS 对应的原始 Chunk
        self.chunks = load_chunks()

        with (STORAGE / "index_config.json").open(
            "r", encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        # 加载已构建的 FAISS 索引
        self.index = faiss.read_index(
            str(STORAGE / "dense.faiss")
        )

        if self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS 索引数量与 Chunk 数量不一致")

        if self.config["chunk_count"] != len(self.chunks):
            raise ValueError("索引配置与 Chunk 数量不一致")

        if self.index.d != self.config["embedding_dimension"]:
            raise ValueError("FAISS 向量维度与配置不一致")

        # E5 模型只加载一次
        self.embedding = E5Embedding(
            self.config["embedding_model"]
        )

        # BM25 语料分词
        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, question, top_k=DENSE_TOP_K):
        """语义向量检索。"""
        if top_k <= 0:
            return []

        query_vector = self.embedding.encode_query(question)

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append(
                format_result(self.chunks[int(idx)], score)
            )

        return results

    def bm25_search(self, question, top_k=BM25_TOP_K):
        """关键词检索，启用中文术语扩展。"""
        if top_k <= 0:
            return []

        query = expand_query(question)
        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        # 分数由高到低排列
        indices = np.argsort(-scores, kind="stable")

        results = []

        for idx in indices:
            if len(results) >= top_k:
                break

            # 排除零分和负分结果
            if scores[idx] <= 0:
                continue

            results.append(
                format_result(self.chunks[int(idx)], scores[idx])
            )

        return results

    def hybrid_search(self, question, top_k=CONTEXT_TOP_K):
        """Dense + BM25 + RRF。"""
        if top_k <= 0:
            return []

        dense = self.dense_search(
            question, top_k=DENSE_TOP_K
        )

        sparse = self.bm25_search(
            question, top_k=BM25_TOP_K
        )

        fused = rrf_fuse(dense, sparse, k=RRF_K)

        return fused[:top_k]
