# 本文件用于测试 WifiAgent 的流程控制与异常处理。
#
# 测试方式：
# 使用 Mock（测试替身）代替真实模型客户端和工具，预先设定模型返回内容、
# 检索结果或异常，再执行真实的 run() 和 _execute()，检查处理结果是否符合预期。
# 不请求真实 API、不加载 E5 检索模型、不读取 PDF。
#
# 主要检查：
# 1. 正常搜索后，能否基于已登记的证据返回带引用的答案。
# 2. 重复获得同一片段时，是否复用原有证据编号。
# 3. 不存在的引用、不支持的引用格式和缺少引用的回答，能否被拦截。
# 4. 工具调用是否受预算限制，包括单条模型消息中的多个调用。
# 5. 非法工具名称、错误参数和工具异常，能否被正确处理并反馈给模型。
# 6. 模型接口异常、最终输出格式错误和请求前超时，能否明确结束任务。
#
# 判断方式：
# assert 用于检查实际结果是否符合预期；条件不成立时，pytest 会报告失败。
# 故意让模拟模型或工具出错后，Agent 正确处理该错误，才算测试通过。
#
# 测试范围：
# 这些测试验证程序规则，不评估真实检索效果、答案正确性、证据是否支持结论，
# 或拒答是否合理；也不验证正在执行的同步工具能否被强制中断。
# 真实任务表现需要通过真实模型、真实资料和人工评审另行验证。
#
# 运行命令（在项目根目录执行）：
# .\.venv\Scripts\python.exe -m pytest tests/test_agent.py -q

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.agent as agent_module
from src.agent import WifiAgent
from src.tools import success


# ============================================================
# 1. 模拟模型返回的消息对象
# ============================================================

class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        # content：模型的文本回答
        # tool_calls：模型提出的工具调用列表
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        # 模拟OpenAI SDK消息对象的model_dump方法。
        # Agent会调用它，将模型消息加入下一轮请求的历史中。
        message = {"role": "assistant"}

        if self.content is not None:
            message["content"] = self.content

        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]

        return message


# ============================================================
# 2. 构造测试需要的工具请求、最终回答和检索结果
# ============================================================

def tool_call(call_id, name="search_wifi_docs", args=None):
    # 构造一个模拟工具请求。
    # SimpleNamespace支持通过call.function.name这样的属性方式访问。
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            # 模型返回的arguments是JSON字符串，而不是Python字典。
            arguments=json.dumps(
                {"query": "OFDMA"} if args is None else args,
                ensure_ascii=False,
            ),
        ),
    )


def final_message(answer, answerable=True):
    # 构造符合Agent约定的最终回答JSON。
    # answerable=True表示作答，False表示拒答。
    return FakeMessage(
        content=json.dumps(
            {"answerable": answerable, "answer": answer},
            ensure_ascii=False,
        )
    )


def chunk_result():
    # 返回一条固定的模拟证据。
    # 每次调用都会创建新的字典，避免测试之间共享可变数据。
    return success(
        {
            "results": [
                {
                    "chunk_id": "test_chunk_a",
                    "score": 0.03,
                    "text": "OFDMA将信道划分为资源单元。",
                    "truncated": False,
                    "metadata": {
                        "doc_id": "test_doc",
                        "title": "Test Wi-Fi Document",
                        "pdf_page": 1,
                        "source_url": "https://example.com/test.pdf",
                    },
                }
            ]
        }
    )


# ============================================================
# 3. 创建供各测试复用的模拟Agent
# ============================================================

@pytest.fixture
def build_agent(monkeypatch):
    # fixture会在需要它的测试运行前执行。
    # 固定时间，避免普通测试因机器速度不同而触发时间预算。
    # monkeypatch会在测试结束后自动恢复被替换的函数。
    monkeypatch.setattr(
        agent_module.time,
        "monotonic",
        lambda: 0.0,
    )

    def build(messages):
        # __new__只创建对象，不执行__init__。
        # 因此不会检查真实API配置，也不会创建真实客户端。
        agent = WifiAgent.__new__(WifiAgent)

        # 使用Mock替代WifiTools，可记录调用次数和参数。
        agent.tools = Mock()

        # 每次模拟搜索都返回一份新的固定证据。
        agent.tools.search_wifi_docs.side_effect = (
            lambda **kwargs: chunk_result()
        )

        # 将预设消息变成迭代器，每次模型请求取下一条。
        responses = iter(messages)

        # 保存Agent发出的模型请求，供测试检查消息内容。
        requests = []

        def create(**kwargs):
            # 深拷贝请求，防止后续messages追加操作改变历史快照。
            requests.append(copy.deepcopy(kwargs))

            message = next(responses)

            # 模拟response.choices[0].message的结构。
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

        # Mock既能执行create，也能记录它是否被调用。
        create_mock = Mock(side_effect=create)

        # 构造与真实SDK相同的属性访问路径。
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_mock)
            )
        )

        return agent, requests

    return build


# ============================================================
# 4. 正常流程：搜索一次，然后基于证据回答
# ============================================================

def test_search_then_answer(build_agent):
    agent, requests = build_agent(
        [
            # 第一轮模型请求搜索工具。
            FakeMessage(tool_calls=[tool_call("call_a")]),

            # 第二轮模型根据工具证据给出最终答案。
            final_message("OFDMA划分资源单元。[S1]"),
        ]
    )

    result = agent.run("OFDMA是什么？")

    # 检查结束状态、调用次数和证据映射。
    assert result["status"] == "answered"
    assert result["tool_calls"] == 1
    assert result["model_rounds"] == 2
    assert result["citations"] == ["S1"]
    assert result["evidence"]["S1"]["chunk_id"] == "test_chunk_a"

    # 第二次模型请求中，应包含第一次工具的执行结果。
    tool_messages = [
        message
        for message in requests[1]["messages"]
        if message["role"] == "tool"
    ]

    # 工具结果必须对应原始调用ID。
    assert tool_messages[0]["tool_call_id"] == "call_a"

    # 返回模型的证据中，应已经加入程序分配的S1编号。
    returned = json.loads(tool_messages[0]["content"])
    assert returned["data"]["results"][0]["evidence_id"] == "S1"


# ============================================================
# 5. 重复检索同一片段时，引用编号保持不变
# ============================================================

def test_repeated_chunk_keeps_same_id(build_agent):
    agent, _ = build_agent(
        [
            FakeMessage(tool_calls=[tool_call("call_a")]),
            FakeMessage(tool_calls=[tool_call("call_b")]),
            final_message("OFDMA划分资源单元。[S1]"),
        ]
    )

    result = agent.run("OFDMA是什么？")

    assert result["status"] == "answered"

    # 两次搜索返回相同chunk_id，证据表中只保留一条。
    assert len(result["evidence"]) == 1

    # 两次工具轨迹都应使用S1，不能第二次变成S2。
    assert result["trace"][0]["evidence_ids"] == ["S1"]
    assert result["trace"][1]["evidence_ids"] == ["S1"]


# ============================================================
# 6. 无效引用检查
# ============================================================

@pytest.mark.parametrize(
    "answer",
    [
        "结论。[S99]",       # 引用了不存在的证据
        "结论。[S1,S99]",    # 使用了不支持的合并格式
        "结论。[S1-S3]",     # 使用了不支持的区间格式
    ],
)
def test_invalid_citation_is_blocked(build_agent, answer):
    # parametrize会针对上面的三种输入分别执行一次测试。
    agent, _ = build_agent(
        [
            FakeMessage(tool_calls=[tool_call("call_a")]),
            final_message(answer),
        ]
    )

    result = agent.run("测试问题")

    assert result["status"] == "invalid_citation"

    # 被拦截的答案不能返回为有效引用结果。
    assert result["citations"] == []


def test_answer_without_evidence_is_blocked(build_agent):
    # 模型没有调用工具，却直接声称引用了S1。
    agent, _ = build_agent(
        [final_message("OFDMA是一种多址技术。[S1]")]
    )

    result = agent.run("OFDMA是什么？")

    # S1并不存在，应作为无效引用拦截。
    assert result["status"] == "invalid_citation"


def test_answer_without_citation_is_blocked(build_agent):
    # 虽然已经取得证据，但模型回答中没有标注来源。
    agent, _ = build_agent(
        [
            FakeMessage(tool_calls=[tool_call("call_a")]),
            final_message("OFDMA划分资源单元。"),
        ]
    )

    result = agent.run("OFDMA是什么？")

    assert result["status"] == "missing_evidence"


# ============================================================
# 7. 工具预算检查：一条消息内的多个调用也要逐个计数
# ============================================================

def test_tool_budget_counts_each_call(build_agent):
    # 模拟模型在同一条消息中请求4次工具调用。
    calls = [tool_call(f"call_{i}") for i in range(4)]

    agent, requests = build_agent(
        [
            FakeMessage(tool_calls=calls),
            final_message("OFDMA划分资源单元。[S1]"),
        ]
    )

    result = agent.run("测试调用预算")

    assert result["status"] == "answered"

    # 最多消耗3次调用预算，真实工具也只能执行3次。
    assert result["tool_calls"] == 3
    assert agent.tools.search_wifi_docs.call_count == 3

    tool_messages = [
        message
        for message in requests[1]["messages"]
        if message["role"] == "tool"
    ]

    # 第4次虽然不执行，仍须回传对应的预算错误。
    # 因此共有4条tool结果消息。
    assert len(tool_messages) == 4

    # 每条结果都必须对应其原始调用ID。
    assert [message["tool_call_id"] for message in tool_messages] == [
        call.id for call in calls
    ]

    denied = json.loads(tool_messages[-1]["content"])
    assert denied["error"]["code"] == "TOOL_BUDGET_EXCEEDED"

    # 下一次模型请求必须禁用工具，让模型回答或拒答。
    assert requests[1]["tool_choice"] == "none"


# ============================================================
# 8. 非法工具名称与参数检查
# ============================================================

@pytest.mark.parametrize(
    ("name", "arguments", "expected_code"),
    [
        # 未登记的工具不能执行。
        ("unknown_tool", "{}", "UNKNOWN_TOOL"),

        # 无法解析的JSON参数。
        ("search_wifi_docs", "{broken", "INVALID_ARGUMENT"),

        # 参数必须是对象，不能是列表。
        ("search_wifi_docs", "[]", "INVALID_ARGUMENT"),

        # 不允许传入工具接口之外的额外字段。
        (
            "search_wifi_docs",
            '{"query":"OFDMA","file_path":"anything"}',
            "INVALID_ARGUMENT",
        ),
    ],
)
def test_invalid_tool_request(
    build_agent, name, arguments, expected_code
):
    call = tool_call("call_bad", name=name)
    call.function.arguments = arguments

    agent, _ = build_agent(
        [
            FakeMessage(tool_calls=[call]),
            final_message("本次未取得有效证据。", answerable=False),
        ]
    )

    result = agent.run("测试非法请求")

    # 无效请求也消耗调用预算，防止模型无限提交错误请求。
    assert result["tool_calls"] == 1

    # 检查轨迹记录了正确错误类型。
    assert result["trace"][0]["error"]["code"] == expected_code

    # 参数校验失败时，不能调用底层真实功能。
    agent.tools.search_wifi_docs.assert_not_called()
    agent.tools.get_pdf_page.assert_not_called()


# ============================================================
# 9. 工具异常需要回传模型，而不是导致程序直接崩溃
# ============================================================

def test_tool_exception_is_returned_to_model(build_agent):
    agent, requests = build_agent(
        [
            FakeMessage(tool_calls=[tool_call("call_a")]),
            final_message("工具执行失败，无法完成取证。", False),
        ]
    )

    # 将工具行为替换为主动抛出异常。
    agent.tools.search_wifi_docs.side_effect = RuntimeError("mock")

    result = agent.run("测试工具异常")

    assert result["status"] == "abstained"
    assert result["trace"][0]["error"]["code"] == "TOOL_FAILED"

    # 检查第二轮模型确实收到了工具执行失败的结果。
    tool_message = next(
        message
        for message in requests[1]["messages"]
        if message["role"] == "tool"
    )
    assert json.loads(tool_message["content"])["ok"] is False


# ============================================================
# 10. 模型接口异常需要明确结束
# ============================================================

def test_model_exception_stops_run(build_agent):
    agent, _ = build_agent([])

    # 模拟API调用抛出异常。
    agent.client.chat.completions.create.side_effect = (
        RuntimeError("mock")
    )

    result = agent.run("测试接口异常")

    assert result["status"] == "model_error"
    assert result["model_rounds"] == 1
    assert result["tool_calls"] == 0


# ============================================================
# 11. 最终输出必须符合约定的JSON结构
# ============================================================

def test_invalid_final_json_is_blocked(build_agent):
    agent, _ = build_agent(
        [FakeMessage(content="这不是约定的JSON")]
    )

    result = agent.run("测试输出格式")

    assert result["status"] == "invalid_output"


# ============================================================
# 12. 时间预算耗尽后，不应继续发送模型请求
# ============================================================

def test_time_budget_expires_before_request(
    build_agent, monkeypatch
):
    agent, _ = build_agent([])

    # 第一次读时间得到0秒，表示任务开始；
    # 第二次得到121秒，超过当前120秒预算。
    clock = iter([0.0, 121.0])

    monkeypatch.setattr(
        agent_module.time,
        "monotonic",
        lambda: next(clock, 121.0),
    )

    result = agent.run("测试时间预算")

    assert result["status"] == "timeout"

    # 确认预算检查发生在发送API请求之前。
    agent.client.chat.completions.create.assert_not_called()
