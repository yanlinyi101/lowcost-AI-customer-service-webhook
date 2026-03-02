"""腾讯云 COS 聊天日志模块"""
import asyncio
import json
import logging
from datetime import datetime

from qcloud_cos import CosConfig, CosS3Client

from config import COS_ENABLED, COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET

logger = logging.getLogger(__name__)

_client: CosS3Client | None = None


def _get_client() -> CosS3Client:
    global _client
    if _client is None:
        config = CosConfig(
            Region=COS_REGION,
            SecretId=COS_SECRET_ID,
            SecretKey=COS_SECRET_KEY,
        )
        _client = CosS3Client(config)
    return _client


def _upload(openid: str, user_text: str, ai_reply: str) -> None:
    """同步上传（在线程池中执行）"""
    now = datetime.now()
    key = f"chat_logs/{now.strftime('%Y-%m-%d')}/{now.strftime('%H-%M-%S')}_{openid[:8]}.json"
    body = json.dumps({
        "time":   now.strftime("%Y-%m-%d %H:%M:%S"),
        "openid": openid[:8] + "...",
        "user":   user_text,
        "ai":     ai_reply,
    }, ensure_ascii=False)

    try:
        _get_client().put_object(
            Bucket=COS_BUCKET,
            Body=body.encode("utf-8"),
            Key=key,
            ContentType="application/json",
        )
        logger.info(f"[COS日志] 写入成功 {key}")
    except Exception as e:
        logger.error(f"[COS日志] 写入失败: {e}")


async def log_chat(openid: str, user_text: str, ai_reply: str) -> None:
    """异步写入聊天日志，不阻塞主流程"""
    if not COS_ENABLED:
        return
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _upload, openid, user_text, ai_reply)
