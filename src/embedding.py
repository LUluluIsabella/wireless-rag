
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small" # 文本 → 384 维向量


class E5Embedding:
    def __init__(self, model_name=MODEL_NAME):
        # 初始化模型，只加载一次
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts, batch_size=32):
        """
        将文档文本编码为向量。
        E5 文档需要 passage: 前缀。
        """
        inputs = ["passage: " + text for text in texts]

        vectors = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.ascontiguousarray(
            vectors, dtype=np.float32
        )

    def encode_query(self, question):
        """
        将用户问题编码为向量。
        E5 查询需要 query: 前缀。
        """
        vectors = self.model.encode(
            ["query: " + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.ascontiguousarray(
            vectors, dtype=np.float32
        )
