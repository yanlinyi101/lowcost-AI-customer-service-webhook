"""
RAG（检索增强生成）模块
- 从 knowledge_base.json 加载知识库
- 根据用户问题检索最相关的条目
- 将结果注入 AI 系统提示词，引导 AI 优先基于知识库回答

知识库格式（knowledge_base.json）：
[
  {
    "question": "如何申请退款？",
    "answer":   "支持7天无理由退款，请在订单页面点击申请退款...",
    "keywords": ["退款", "退货", "退钱", "不想要", "七天"]
  },
  ...
]
"""

import json
import logging
from pathlib import Path

from config import RAG_TOP_K, RAG_MIN_SCORE

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent / "knowledge_base.json"

# ──────────────────────────────────────────
# 加载知识库
# ──────────────────────────────────────────

def load_knowledge_base() -> list[dict]:
    if not KB_PATH.exists():
        logger.warning(f"知识库文件不存在：{KB_PATH}")
        return []
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        logger.info(f"知识库加载成功，共 {len(kb)} 条")
        return kb
    except Exception as e:
        logger.error(f"知识库加载失败：{e}")
        return []


# 启动时加载一次，避免每次请求都读文件
_kb_cache: list[dict] = []

def get_kb() -> list[dict]:
    global _kb_cache
    if not _kb_cache:
        _kb_cache = load_knowledge_base()
    return _kb_cache

def reload_kb() -> None:
    """手动重新加载知识库（更新知识库后调用）"""
    global _kb_cache
    _kb_cache = load_knowledge_base()


# ──────────────────────────────────────────
# 相关性评分（关键词匹配 + 汉字字符重叠）
# ──────────────────────────────────────────

def _score(query: str, entry: dict) -> float:
    """
    计算用户问题与知识库条目的相关性分数。
    策略：
    1. 关键词命中：用户问题包含关键词，每命中一个 +2 分
    2. 问题字符重叠：与 question 字段共享的汉字数量，每个 +0.5 分
    """
    score = 0.0

    # 1. 关键词匹配（权重最高）
    for kw in entry.get("keywords", []):
        if kw and kw in query:
            score += 2.0

    # 2. question 字段汉字字符重叠
    question = entry.get("question", "")
    for char in query:
        if "\u4e00" <= char <= "\u9fff" and char in question:
            score += 0.5

    return score


# ──────────────────────────────────────────
# 检索入口
# ──────────────────────────────────────────

def retrieve(query: str) -> tuple[str, list[str]]:
    """
    根据用户问题检索知识库。
    返回：
      - context_text: 注入 AI 提示词的参考内容（str）
      - image_urls:   命中条目中的图片链接列表（list[str]），可为空
    """
    kb = get_kb()
    if not kb:
        return "", []

    scored = [
        (entry, _score(query, entry))
        for entry in kb
    ]
    scored = [(e, s) for e, s in scored if s >= RAG_MIN_SCORE]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:RAG_TOP_K]
    if not top:
        return "", []

    lines = []
    image_urls = []
    for entry, score in top:
        lines.append(f"问：{entry['question']}\n答：{entry['answer']}")
        logger.debug(f"[RAG] 命中 score={score:.1f} | {entry['question'][:20]}")
        # 收集图片链接（只取非空值，去重保持顺序）
        url = entry.get("image_url", "").strip()
        if url and url not in image_urls:
            image_urls.append(url)

    return "\n\n---\n\n".join(lines), image_urls
