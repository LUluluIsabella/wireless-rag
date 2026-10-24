# 这是本节的核心脚本。它会完成以下工作：
# 1. 读取 chunks.jsonl。
# 2. 调用 E5Embedding 为所有 Chunk 生成向量。
# 3. 创建 FAISS 索引。
# 4. 保存索引和原始 Chunk 的对应关系。
# 5. 保存模型配置与语料校验信息。


import json
import hashlib
from pathlib import Path

import faiss
import numpy as np

from src.embedding import E5Embedding, MODEL_NAME
from src.chunking import CHUNK_SIZE, CHUNK_OVERLAP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"
STORAGE_DIR = PROJECT_ROOT / "storage"


def load_chunks():
    """读取第 8 节生成的 Chunk。"""

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("chunks.jsonl 为空")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("发现重复的 chunk_id")

    return chunks


def calculate_sha256(path):
    """计算语料文件的 SHA256 校验值。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    print("========== Step 1: 读取 Chunks ==========")

    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunk 总数: {len(chunks)}")

    print("\n========== Step 2: Embedding ==========")

    embedding = E5Embedding()

    # 检查完整模型输入是否超出长度限制
    tokenizer = embedding.model.tokenizer
    max_length = embedding.model.max_seq_length

    for chunk, text in zip(chunks, texts):
        n = len(tokenizer.encode(
            "passage: " + text,
            add_special_tokens=True,
        ))

        if n > max_length:
            raise ValueError(
                f"{chunk['chunk_id']} 输入长度 {n} "
                f"超过模型上限 {max_length}"
            )

    vectors = embedding.encode_documents(texts)

    print("向量矩阵 Shape:", vectors.shape)

    assert vectors.ndim == 2
    assert vectors.shape[0] == len(chunks)
    assert vectors.shape[1] == 384
    assert np.isfinite(vectors).all()

    # 检查归一化是否成功
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    print("\n========== Step 3: 创建 FAISS 索引 ==========")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print("索引向量数量:", index.ntotal)

    assert index.ntotal == len(chunks)

    print("\n========== Step 4: 保存索引 ==========")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index, str(STORAGE_DIR / "dense.faiss")
    )

    # 这个文件的行顺序必须与 FAISS 向量顺序一致
    with (STORAGE_DIR / "chunks.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for chunk in chunks:
            f.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # 保存索引构建参数
    config = {
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dimension,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "corpus_sha256": calculate_sha256(CHUNKS_PATH),
        "index_type": "IndexFlatIP",
        "normalized": True,
    }

    with (STORAGE_DIR / "index_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("索引保存成功！")
    print("存储位置:", STORAGE_DIR)


if __name__ == "__main__":
    main()
