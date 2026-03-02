"""
AI 服务模块
- 调用 DeepSeek API（兼容 OpenAI 格式）
- 维护每个用户的对话历史
- RAG：检索知识库，将相关内容注入提示词
- 支持更换 AI 提供商（改 BASE_URL + MODEL 即可）
"""

import asyncio
import logging
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

logger = logging.getLogger(__name__)

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
    context = ""
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

    reply = ""
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(base_url=AI_BASE_URL, timeout=60) as client:
                resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
            break
        except Exception as e:
            if attempt == 0:
                logger.warning(f"[AI] 第1次失败，1s后重试: {type(e).__name__}: {e}")
                await asyncio.sleep(1)
            else:
                logger.error(f"[AI] 调用失败(已重试): {type(e).__name__}: {e}")
                # 有知识库命中时，直接用第一条答案兜底，同时保留图片
                if context:
                    first_entry = context.split("\n\n---\n\n")[0]
                    reply = first_entry.split("答：", 1)[1].strip() if "答：" in first_entry else context
                    logger.info("[AI] 使用知识库答案兜底")
                else:
                    reply = "抱歉，我暂时无法回复，请稍后再试或联系人工客服。"
                    image_urls = []

    add_to_history(openid, "assistant", reply)
    return reply, image_urls
