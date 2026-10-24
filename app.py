
"""
Wireless-RAG Streamlit Demo

功能：
1. 用户输入 Wi-Fi 6 技术问题
2. 选择 Dense / BM25 / Hybrid 检索模式
3. 调用 RAG Pipeline 生成答案
4. 展示引用来源与 PDF 原文
5. 缓存检索模型
6. 处理 API 异常
"""

import time

import streamlit as st

from src.pipeline import RAGPipeline


# =========================================
# 1. 页面配置
# =========================================

st.set_page_config(
    page_title="Wireless-RAG",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Wireless-RAG")

st.caption(
    "面向 Wi-Fi 6 技术文档的可溯源问答系统"
)

st.divider()


# =========================================
# 2. 加载 RAG Pipeline
# =========================================

@st.cache_resource
def load_pipeline():
    """
    缓存 RAG Pipeline。

    Streamlit 每次交互都会重新运行脚本。
    如果不缓存，就会反复加载 Embedding 模型、
    FAISS 索引及 BM25。
    """
    return RAGPipeline()


# =========================================
# 3. 侧边栏：检索参数
# =========================================

with st.sidebar:

    st.header("⚙️ 检索设置")

    mode = st.selectbox(
        "检索模式",
        options=["hybrid", "dense", "bm25"],
        index=0,
        format_func=lambda x: {
            "hybrid": "Hybrid (RRF)",
            "dense": "Dense (FAISS)",
            "bm25": "BM25",
        }[x],
    )

    top_k = st.slider(
        "检索证据数量 Top-K",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.markdown("### 系统说明")

    st.write(
        "系统使用 Wi-Fi 6 技术文档作为知识库，"
        "通过检索增强生成提供带引用的回答。"
    )

    st.caption(
        "引用编号由程序映射到原始文档，"
        "不由 LLM 自行生成页码或 URL。"
    )


# =========================================
# 4. 用户问题输入
# =========================================

st.subheader("💬 技术问答")

example_questions = [
    "自定义问题",
    "OFDMA 和 MU-MIMO 有什么区别？",
    "20 MHz 信道最多容纳多少个 26-tone RU？",
    "为什么 1024-QAM 需要更高 SNR？",
    "TWT 如何帮助 Wi-Fi 6 设备节能？",
]

selected_example = st.selectbox(
    "选择示例问题",
    options=example_questions,
)

default_question = (
    ""
    if selected_example == "自定义问题"
    else selected_example
)

question = st.text_area(
    "请输入问题",
    value=default_question,
    height=100,
    placeholder="例如：Wi-Fi 6 中 OFDMA 的作用是什么？",
)

ask_button = st.button(
    "🔍 开始查询",
    type="primary",
    use_container_width=True,
)


# =========================================
# 5. 执行 RAG 查询
# =========================================

if ask_button:

    if not question.strip():
        st.warning("请输入一个问题。")
        st.stop()

    try:
        with st.spinner("正在加载模型并检索文档..."):

            pipeline = load_pipeline()

            start_time = time.perf_counter()

            result = pipeline.query(
                question=question,
                mode=mode,
                top_k=top_k,
            )

            elapsed = time.perf_counter() - start_time

    except Exception as e:
        st.error(
            "查询失败，请检查模型 API、网络连接及本地索引。"
        )
        st.exception(e)
        st.stop()

    # =====================================
    # 6. 展示答案
    # =====================================

    st.divider()

    st.subheader("🤖 AI 回答")

    answer = result.get("answer", "")

    if answer:
        st.markdown(answer)
    else:
        st.warning("模型没有返回答案。")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "检索模式",
            mode.upper(),
        )

    with col2:
        st.metric(
            "有效引用数量",
            len(result.get("citations", [])),
        )

    with col3:
        st.metric(
            "端到端耗时",
            f"{elapsed:.2f} s",
        )

    # =====================================
    # 7. 检查无效引用
    # =====================================

    invalid_citations = result.get(
        "invalid_citations", []
    )

    if invalid_citations:
        st.error(
            "检测到无法验证的引用编号："
            + ", ".join(invalid_citations)
        )

    # =====================================
    # 8. 展示文档来源
    # =====================================

    st.divider()

    st.subheader("📚 引用来源")

    sources = result.get("sources", [])

    if not sources:
        st.info(
            "本次回答没有可展示的有效引用来源。"
        )

    for source in sources:

        source_id = source.get(
            "source_id", "Unknown"
        )

        chunk_id = source.get(
            "chunk_id", "Unknown"
        )

        metadata = source.get(
            "metadata", {}
        )

        title = metadata.get(
            "title", "未知文档"
        )

        pdf_page = metadata.get(
            "pdf_page", "未知"
        )

        source_url = metadata.get(
            "source_url", ""
        )

        chunk_text = source.get(
            "text", ""
        )

        with st.expander(
            f"[{source_id}] {title} · PDF 第 {pdf_page} 页"
        ):

            st.write(
                f"**Chunk ID：** {chunk_id}"
            )

            st.write(
                f"**文档标题：** {title}"
            )

            st.write(
                f"**PDF 页序号：** {pdf_page}"
            )

            if source_url:
                st.link_button(
                    "📄 查看原始 PDF",
                    source_url,
                )

            st.markdown("**原始证据：**")

            st.text(chunk_text)

    st.divider()

    st.caption(
        "回答由 LLM 基于检索文档生成。"
        "请结合原始引用证据核验重要技术结论。"
    )
