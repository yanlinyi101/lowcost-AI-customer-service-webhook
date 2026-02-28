"""
AI 服务模块
- 调用 DeepSeek API（兼容 OpenAI 格式）
- 维护每个用户的对话历史
- RAG：检索知识库，将相关内容注入提示词
- 支持更换 AI 提供商（改 BASE_URL + MODEL 即可）
"""

from collections import defaultdict, deque

import httpx

from config import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_MODEL,
    SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    HUMAN_TAKEOVER_KEYWORDS,
    RAG_ENABLED,
)
from rag_service import retrieve

# ──────────────────────────────────────────
# 每个用户的对话历史（内存存储，重启清空）
# key: openid, value: deque of {"role": ..., "content": ...}
# ──────────────────────────────────────────
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))


def needs_human(text: str) -> bool:
    """检测用户是否请求转人工"""
    return any(kw in text for kw in HUMAN_TAKEOVER_KEYWORDS)


def add_to_history(openid: str, role: str, content: str) -> None:
    _history[openid].append({"role": role, "content": content})


def get_history(openid: str) -> list:
    return list(_history[openid])


def clear_history(openid: str) -> None:
    _history[openid].clear()


# ──────────────────────────────────────────
# 调用 AI 生成回复
# ──────────────────────────────────────────

async def get_ai_reply(openid: str, user_message: str) -> tuple[str, list[str]]:
    """
    调用 AI 接口获取回复。
    返回：
      - reply_text: AI 生成的文字回复
      - image_urls: 知识库命中条目中附带的图片链接（可为空列表）
    """
    add_to_history(openid, "user", user_message)

    # RAG：检索知识库
    system = SYSTEM_PROMPT
    image_urls: list[str] = []
    if RAG_ENABLED:
        context, image_urls = retrieve(user_message)
        if context:
            system += (
                "\n\n【知识库参考信息】\n"
                "以下是与用户问题相关的官方信息，请优先基于此回答，"
                "不要与之矛盾，也不要编造额外内容：\n\n"
                + context
            )

    messages = [
        {"role": "system", "content": system},
        *get_history(openid),
    ]

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(base_url=AI_BASE_URL, timeout=30) as client:
            resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
            data = resp.json()

        reply = data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"[AI] 调用失败: {e}")
        reply = "抱歉，我暂时无法回复，请稍后再试或联系人工客服。"

    add_to_history(openid, "assistant", reply)
    return reply, image_urls
