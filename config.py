import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 微信小程序配置（在 mp.weixin.qq.com 后台获取）
# ─────────────────────────────────────────────
WECHAT_TOKEN            = os.getenv("WECHAT_TOKEN", "")
WECHAT_APP_ID           = os.getenv("WECHAT_APP_ID", "")
WECHAT_APP_SECRET       = os.getenv("WECHAT_APP_SECRET", "")
WECHAT_ENCODING_AES_KEY = os.getenv("WECHAT_ENCODING_AES_KEY", "")

# ─────────────────────────────────────────────
# AI 配置（默认使用 DeepSeek，可切换）
# ─────────────────────────────────────────────
AI_API_KEY   = os.getenv("AI_API_KEY", "")
AI_BASE_URL  = os.getenv("AI_BASE_URL", "https://api.deepseek.com")
AI_MODEL     = os.getenv("AI_MODEL", "deepseek-chat")

# ─────────────────────────────────────────────
# 客服 AI 系统提示词（根据你的业务修改）
# ─────────────────────────────────────────────
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
你是一名专业、亲切的在线客服助手。请遵守以下规则：
1. 用简洁、友好的语气回复，每次回复不超过200字
2. 如果用户询问退款、投诉等敏感问题，告知会转接人工客服
3. 不要编造产品信息，不确定时说"我帮您确认后回复"
4. 始终保持礼貌，称呼用户为"您"
""".strip())

# ─────────────────────────────────────────────
# 触发人工接管的关键词
# ─────────────────────────────────────────────
HUMAN_TAKEOVER_KEYWORDS = ["转人工", "人工客服", "人工", "转接", "真人"]

# 转接的指定客服账号（留空则自动分配）格式：xxx@公众号ID
KF_ACCOUNT = os.getenv("KF_ACCOUNT", "")

# 每个用户保留的对话历史轮数（太多会增加 AI 费用）
MAX_HISTORY_TURNS = 5

# ─────────────────────────────────────────────
# RAG 知识库配置
# ─────────────────────────────────────────────
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_TOP_K   = int(os.getenv("RAG_TOP_K", "3"))   # 每次检索返回最相关的条目数
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "1.0"))  # 最低相关性分数，低于此分数不注入

# ─────────────────────────────────────────────
# 腾讯云 COS 聊天日志配置
# ─────────────────────────────────────────────
COS_ENABLED    = os.getenv("COS_ENABLED", "false").lower() == "true"
COS_SECRET_ID  = os.getenv("COS_SECRET_ID", "")
COS_SECRET_KEY = os.getenv("COS_SECRET_KEY", "")
COS_REGION     = os.getenv("COS_REGION", "ap-guangzhou")
COS_BUCKET     = os.getenv("COS_BUCKET", "")    # 格式：bucketname-appid
