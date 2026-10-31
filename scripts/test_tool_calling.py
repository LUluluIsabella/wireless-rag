# 用途：验证当前模型的最小工具调用链路。
# 流程：模型请求工具 → 校验参数 → 执行只读函数 → 回传结果 → 生成答案。
# 本脚本不加载检索器，不读取PDF，不记录密钥。
# 最多执行一次工具调用；每次API请求超时30秒，关闭自动重试。

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_project_info() -> dict:
    return {
        "project": "wireless-rag",
        "scope": "Wi-Fi 6技术文档问答",
        "retrieval": "Dense + BM25 + RRF",
        "agent_status": "最小工具调用验证阶段",
    }


TOOLS = [
    {
        "type": "function", # 说明这是一个函数工具
        "function": {
            "name": "get_project_info", # 工具名称, 模型请求调用时使用的名字
            "description": "读取本项目的真实功能范围和当前开发状态。", # 帮助模型判断什么时候应该使用它
            "parameters": { # 说明允许传入什么参数
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]


def main() -> None:
    config = {
        name: os.getenv(name, "").strip()
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    }

    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(f"缺少配置：{', '.join(missing)}")

    client = OpenAI(
        api_key=config["LLM_API_KEY"],
        base_url=config["LLM_BASE_URL"],
        timeout=30.0,
        max_retries=0,
    )

    messages = [ # 准备发送给模型的消息
        {
            "role": "system",
            "content": (
                "你是项目助手。涉及项目当前状态时必须先查询工具。"
                "仅根据工具返回的信息回答，不推测已经完成的功能。"
            ),
        },
        {
            "role": "user",
            "content": "请查询本项目的功能范围和当前Agent开发状态。" # 类似于 用户发送的prompt
        },
    ]  # 此时 messages 中没有工具返回的项目信息，所以模型需要通过工具获取。

    first = client.chat.completions.create( # 第一次 API 请求：模型提出工具调用
    # Python 程序通过 client 调用 API，把请求发送给服务端指定的模型。
        model=config["LLM_MODEL"],
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    message = first.choices[0].message
    calls = message.tool_calls or []

    if len(calls) != 1:
        raise RuntimeError(
            f"预期一次工具调用，实际为 {len(calls)} 次。"
            "请检查模型是否支持工具调用及提示词是否生效。"
        )

    call = calls[0]

    if call.type != "function":
        raise ValueError("不支持的工具调用类型")

    if call.function.name != "get_project_info":
        raise ValueError("模型请求了未知工具")

    arguments = json.loads(call.function.arguments)

    if not isinstance(arguments, dict) or arguments:
        raise ValueError("get_project_info只接受空对象参数")

    print("工具名称：", call.function.name)
    print("调用ID：", call.id)
    print("工具参数：", arguments)

    result = get_project_info() # python真正执行的工具

    print("工具结果：", json.dumps(result, ensure_ascii=False))

    messages.append(message.model_dump(exclude_none=True))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )

    final = client.chat.completions.create( # 第二次 API 请求：根据工具结果生成答案
        model=config["LLM_MODEL"],
        messages=messages,
        tools=TOOLS,
        tool_choice="none", #要求这一次不再请求工具，直接生成回答。
    )

    answer = final.choices[0].message.content
    if not answer:
        raise RuntimeError("模型未返回最终答案")

    print("\n最终回答：")
    print(answer)


if __name__ == "__main__":
    main()
