# 用途：为后续Agent提供两个只读工具，当前可独立调用。
# search_wifi_docs：复用现有HybridRetriever，返回片段与真实来源。
# get_pdf_page：只读取sources.csv登记且位于data/raw内的PDF。
# 参数先校验再执行；错误使用统一结构返回。
# 正文有长度限制，截断时明确标记；本文件不调用LLM。

import csv
import logging

from src.config import ROOT_DIR


LOGGER = logging.getLogger(__name__)

MAX_QUERY_CHARS = 1000
MAX_TOP_K = 5
MAX_CHUNK_CHARS = 4000
MAX_PAGE_CHARS = 12000


def success(data: dict) -> dict:
    return {"ok": True, "data": data, "error": None}


def failure(code: str, message: str) -> dict:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
    }


class WifiTools:
    def __init__(self):
        self._retriever = None
        self.sources_path = ROOT_DIR / "data" / "sources.csv"
        self.pdf_root = (ROOT_DIR / "data" / "raw").resolve()

    def _get_retriever(self):
        if self._retriever is None:
            from src.retriever import HybridRetriever

            self._retriever = HybridRetriever()

        return self._retriever

    def _load_sources(self) -> dict:
        sources = {}

        with self.sources_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            required = {"doc_id", "local_file", "title", "source_url"}

            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("资料清单缺少必要字段")

            for row in reader:
                doc_id = row["doc_id"].strip()

                if not doc_id or doc_id in sources:
                    raise ValueError("资料清单存在空或重复doc_id")

                sources[doc_id] = row

        return sources

    def search_wifi_docs(
        self, query: str, top_k: int = 5
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            return failure("INVALID_ARGUMENT", "query必须是非空字符串")

        query = query.strip()

        if len(query) > MAX_QUERY_CHARS:
            return failure(
                "INVALID_ARGUMENT",
                f"query不能超过{MAX_QUERY_CHARS}个字符",
            )

        if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
            return failure(
                "INVALID_ARGUMENT",
                f"top_k必须是1到{MAX_TOP_K}之间的整数",
            )

        try:
            retriever = self._get_retriever()
            results = retriever.hybrid_search(query, top_k=top_k)

            output = []

            for item in results:
                text = item["text"]

                output.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "score": float(item["score"]),
                        "text": text[:MAX_CHUNK_CHARS],
                        "truncated": len(text) > MAX_CHUNK_CHARS,
                        "metadata": dict(item["metadata"]),
                    }
                )

            return success({"results": output})

        except Exception:
            LOGGER.exception("Hybrid检索执行失败")
            return failure(
                "SEARCH_FAILED",
                "检索失败，请检查索引、语料和模型加载状态",
            )

    def get_pdf_page(
        self, doc_id: str, pdf_page: int
    ) -> dict:
        if not isinstance(doc_id, str) or not doc_id.strip():
            return failure("INVALID_ARGUMENT", "doc_id必须是非空字符串")

        if type(pdf_page) is not int or pdf_page < 1:
            return failure(
                "INVALID_ARGUMENT",
                "pdf_page必须是从1开始的整数",
            )

        doc_id = doc_id.strip()

        try:
            sources = self._load_sources()
        except Exception:
            LOGGER.exception("读取资料清单失败")
            return failure(
                "SOURCE_CONFIG_ERROR",
                "资料清单读取失败，请检查sources.csv",
            )

        if doc_id not in sources:
            return failure("DOC_NOT_FOUND", "doc_id未在资料清单中登记")

        source = sources[doc_id]

        try:
            local_file = source["local_file"].strip()

            if not local_file:
                return failure(
                    "SOURCE_CONFIG_ERROR", "该文档未配置local_file"
                )

            pdf_path = (ROOT_DIR / local_file).resolve()

            if not pdf_path.is_relative_to(self.pdf_root):
                return failure(
                    "ACCESS_DENIED", "文档路径超出允许的PDF目录"
                )

            if pdf_path.suffix.lower() != ".pdf":
                return failure("ACCESS_DENIED", "只允许读取PDF文件")

            if not pdf_path.is_file():
                return failure(
                    "PDF_NOT_FOUND", "已登记文档的本地PDF不存在"
                )

            import fitz

            with fitz.open(str(pdf_path)) as document:
                if document.needs_pass:
                    return failure(
                        "PDF_ENCRYPTED", "暂不支持需要密码的PDF"
                    )

                page_count = len(document)

                if pdf_page > page_count:
                    return failure(
                        "PAGE_OUT_OF_RANGE",
                        f"页码超出范围，该文档共{page_count}页",
                    )

                text = document[pdf_page - 1].get_text(
                    "text", sort=True
                ).strip()

            if not text:
                return failure(
                    "EMPTY_PAGE",
                    "该页未提取到文本，可能是空白页或扫描页",
                )

            return success(
                {
                    "page_id": f"{doc_id}:page:{pdf_page}",
                    "text": text[:MAX_PAGE_CHARS],
                    "truncated": len(text) > MAX_PAGE_CHARS,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": source["title"],
                        "source_url": source["source_url"],
                        "document_version": source.get(
                            "document_version", ""
                        ),
                        "pdf_page": pdf_page,
                        "page_count": page_count,
                    },
                }
            )

        except Exception:
            LOGGER.exception("读取PDF页面失败")
            return failure(
                "PDF_READ_FAILED",
                "PDF读取失败，请检查文件和PDF解析依赖",
            )
