from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
)

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


if not LLM_API_KEY:
    raise SystemExit("缺少 LLM_API_KEY，请检查项目根目录的 .env 文件。")

if not LLM_MODEL:
    raise SystemExit("缺少 LLM_MODEL，请检查项目根目录的 .env 文件。")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=30.0,
)

try:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": "请只回复：API调用成功"}
        ],
        max_tokens=64,
    )
    print("模型回复：", response.choices[0].message.content)

except AuthenticationError:
    print("认证失败：请检查 .env 中的 API Key 是否正确。")

except APIConnectionError:
    print("连接失败：请检查网络和 LLM_BASE_URL。")

except APIStatusError as exc:
    print(f"API 返回错误，HTTP 状态码：{exc.status_code}")
    print("错误详情：", exc.response.text)
    print(
        "Trace ID：",
        exc.response.headers.get("x-request-id", "未提供"),
    )