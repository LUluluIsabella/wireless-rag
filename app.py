"""
Wireless-RAG Streamlit Demo

功能：
1. 支持 RAG 与 Agent 两种问答模式。
2. RAG 模式支持 Dense / BM25 / Hybrid 检索。
3. Agent 模式展示状态、工具调用次数、模型轮数及工具轨迹。
4. 展示有效引用及原始证据。
5. 缓存后端资源，并通过锁避免共享实例同时执行查询。
6. 使用 session_state 保存结果，页面交互后仍可查看。
7. 支持下载本次问答的完整 JSON 记录。
8. 只有点击查询按钮时才请求模型。
"""

import json
import time
from threading import Lock
from urllib.parse import urlparse

import streamlit as st

from src.agent import WifiAgent
from src.pipeline import RAGPipeline


# =========================================
# 1. 页面配置与资源加载
# =========================================

st.set_page_config(
    page_title="Wireless-RAG",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Wireless-RAG")
st.caption("面向 Wi-Fi 6 技术文档的可溯源问答系统")


@st.cache_resource
def load_pipeline():
    return RAGPipeline(), Lock()


@st.cache_resource
def load_agent():
    return WifiAgent(), Lock()


# =========================================
# 2. 来源展示
# Agent从evidence中提取实际引用，RAG沿用sources。
# 未引用的Agent证据在单独的折叠区域中查看。
# =========================================

def show_source(source_id, source):
    metadata = source.get("metadata") or {}

    title = metadata.get("title") or source.get(
        "title", "未知文档"
    )
    page = metadata.get("pdf_page") or source.get(
        "pdf_page", source.get("page", "未知")
    )
    url = metadata.get("source_url") or source.get(
        "source_url", ""
    )

    with st.expander(
        f"[{source_id}] {title} · PDF 第 {page} 页"
    ):
        st.write("证据标识：", source.get("chunk_id", "整页证据"))
        st.write("文档标题：", title)
        st.write("PDF 页序号：", page)

        if isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                st.link_button("📄 查看原始 PDF", url)

        st.markdown("**原始证据：**")
        st.text(source.get("text", ""))

        with st.expander("证据完整字段"):
            st.json(source)


# =========================================
# 3. 侧边栏
# =========================================

with st.sidebar:
    st.header("⚙️ 问答设置")

    app_mode = st.radio(
        "问答模式",
        ["RAG", "Agent"],
        horizontal=True,
    )

    retrieval_mode = "hybrid"
    top_k = 5

    if app_mode == "RAG":
        retrieval_mode = st.selectbox(
            "检索模式",
            ["hybrid", "dense", "bm25"],
            format_func=lambda value: {
                "hybrid": "Hybrid (RRF)",
                "dense": "Dense (FAISS)",
                "bm25": "BM25",
            }[value],
        )

        top_k = st.slider(
            "检索证据数量 Top-K",
            min_value=1,
            max_value=10,
            value=5,
        )
    else:
        st.info(
            "Agent 根据问题选择搜索文档或补读 PDF 页面，"
            "检索参数与执行预算沿用后端配置。"
        )

    st.divider()
    st.markdown("### 系统说明")
    st.write(
        "基于 Wi-Fi 6 技术文档回答问题，"
        "支持查看引用证据及 Agent 工具执行记录。"
    )
    st.caption(
        "引用编号由程序映射到证据。"
        "引用有效性检查不能代替技术结论核验。"
    )


# =========================================
# 4. 问题输入
# =========================================

st.subheader("💬 技术问答")

examples = [
    "自定义问题",
    "OFDMA 和 MU-MIMO 有什么区别？",
    "20 MHz 信道最多容纳多少个 26-tone RU？",
    "为什么 1024-QAM 需要更高 SNR？",
    "TWT 如何帮助 Wi-Fi 6 设备节能？",
    "我家路由器昨晚八点的实际平均吞吐量是多少？",
]

selected = st.selectbox("选择示例问题", examples)

# 示例变化时更新输入框；普通页面重跑保留用户编辑。
if st.session_state.get("selected_example") != selected:
    st.session_state["selected_example"] = selected
    st.session_state["question_input"] = (
        "" if selected == "自定义问题" else selected
    )

question = st.text_area(
    "请输入问题",
    key="question_input",
    height=100,
    placeholder="例如：Wi-Fi 6 中 OFDMA 的作用是什么？",
)

submitted = st.button(
    "🔍 开始查询",
    type="primary",
    use_container_width=True,
)


# =========================================
# 5. 执行查询并保存结果
# 总耗时包括首次资源加载及等待执行锁的时间。
# =========================================

if submitted:
    query = question.strip()

    if not query:
        st.warning("请输入一个问题。")
    else:
        # 新任务开始时清除旧结果，避免失败后误显示旧答案。
        st.session_state.pop("last_run", None)

        try:
            started = time.perf_counter()

            with st.spinner("正在检索文档并生成回答……"):
                if app_mode == "Agent":
                    backend, lock = load_agent()

                    with lock:
                        result = backend.run(query)
                else:
                    backend, lock = load_pipeline()

                    with lock:
                        result = backend.query(
                            question=query,
                            mode=retrieval_mode,
                            top_k=top_k,
                        )

            st.session_state["last_run"] = {
                "question": query,
                "mode": app_mode,
                "retrieval_mode": (
                    retrieval_mode if app_mode == "RAG" else None
                ),
                "top_k": top_k if app_mode == "RAG" else None,
                "elapsed_seconds": time.perf_counter() - started,
                "result": result,
            }

        except Exception:
            st.error(
                "查询失败，请检查模型 API、网络连接及本地索引。"
                "详细错误请查看启动 Streamlit 的终端。"
            )
            import logging

            logging.getLogger(__name__).exception(
                "Streamlit query failed"
            )


# =========================================
# 6. 展示已保存结果
# 使用运行时保存的模式，避免切换侧边栏后误标旧结果。
# =========================================

saved = st.session_state.get("last_run")

if saved:
    result = saved["result"]
    is_agent = saved["mode"] == "Agent"

    st.divider()
    st.caption(f"本次结果模式：{saved['mode']}")
    st.write("**本次问题：**", saved["question"])
    st.subheader("🤖 AI 回答")

    status = result.get("status", "")

    if is_agent:
        if status == "answered":
            st.success("已完成回答")
        elif status == "abstained":
            st.warning(
                "本次未作答，请结合下方说明和工具记录查看原因。"
            )
        else:
            st.error(f"本次未正常完成，状态：{status}")

    # 被校验拦截的输出不作为正式答案展示。
    if not is_agent or status in {"answered", "abstained"}:
        answer = result.get("answer", "")
        if answer:
            st.markdown(answer)
        else:
            st.info("本次没有返回答案。")

    if is_agent:
        cols = st.columns(4)
        cols[0].metric("状态", status)
        cols[1].metric(
            "工具调用次数", result.get("tool_calls", 0)
        )
        cols[2].metric(
            "模型请求轮数", result.get("model_rounds", 0)
        )
        cols[3].metric(
            "页面查询总耗时",
            f"{saved['elapsed_seconds']:.2f} s",
        )
    else:
        cols = st.columns(3)
        cols[0].metric(
            "检索模式", saved["retrieval_mode"].upper()
        )
        cols[1].metric(
            "引用数量", len(result.get("citations", []))
        )
        cols[2].metric(
            "页面查询总耗时",
            f"{saved['elapsed_seconds']:.2f} s",
        )

    invalid = result.get("invalid_citations", [])
    if invalid:
        st.error(
            "检测到无法验证的引用编号："
            + ", ".join(map(str, invalid))
        )

    st.divider()
    st.subheader("📚 引用来源")

    if is_agent:
        evidence = result.get("evidence") or {}
        citations = result.get("citations") or []
        cited_ids = list(dict.fromkeys(citations))

        valid_ids = [
            source_id
            for source_id in cited_ids
            if source_id in evidence
        ]

        if not valid_ids:
            st.info("本次没有可展示的有效引用来源。")

        for source_id in valid_ids:
            show_source(source_id, evidence[source_id])

        unused = {
            source_id: source
            for source_id, source in evidence.items()
            if source_id not in cited_ids
        }
        if unused:
            with st.expander("已获取但未引用的证据"):
                st.json(unused)

        st.divider()
        st.subheader("🛠️ 工具执行记录")

        trace = result.get("trace") or []

        if not trace:
            st.info("本次没有工具执行记录。")

        for index, entry in enumerate(trace, start=1):
            # 完整展示原始记录，保留参数、耗时与错误字段。
            tool_name = (
                entry.get("tool")
                or entry.get("name")
                or entry.get("tool_name")
                or "执行记录"
            )
            label = (
                "失败" if entry.get("error") else "查看详情"
            )

            with st.expander(
                f"{index}. {tool_name} · {label}"
            ):
                st.json(entry)

    else:
        sources = result.get("sources") or []

        if not sources:
            st.info("本次没有可展示的有效引用来源。")

        for source in sources:
            show_source(
                source.get("source_id", "Unknown"),
                source,
            )

    st.divider()

    st.download_button(
        "⬇️ 下载本次运行记录",
        data=json.dumps(
            saved,
            ensure_ascii=False,
            indent=2,
        ),
        file_name="wireless_rag_run.json",
        mime="application/json",
    )

    st.caption(
        "回答由模型基于文档生成，"
        "请结合原始证据核验重要技术结论。"
    )
