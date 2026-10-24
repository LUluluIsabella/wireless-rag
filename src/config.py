import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：wireless-rag/
ROOT_DIR = Path(__file__).resolve().parent.parent

# 读取项目根目录下的 .env 文件
load_dotenv(ROOT_DIR / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)

DENSE_TOP_K = 20
BM25_TOP_K = 20
CONTEXT_TOP_K = 5
RRF_K = 60

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def require_llm_config() -> None:
    """在需要调用大模型时检查 API 配置。"""
    if not LLM_API_KEY:
        raise ValueError("请先在项目根目录的 .env 文件中填写 LLM_API_KEY。")
    if not LLM_MODEL:
        raise ValueError("请先在项目根目录的 .env 文件中填写 LLM_MODEL。")