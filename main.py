"""
微信小程序 AI 客服 Webhook 服务
FastAPI 主入口

本地启动命令：
    uvicorn main:app --host 0.0.0.0 --port 8000

腾讯云函数（SCF Web函数）启动：
    由 scf_bootstrap 自动启动，端口固定为 9000
"""

import logging

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, Response

from ai_service import get_ai_reply, needs_human, clear_history
from cos_logger import log_chat
from config import WECHAT_ENCODING_AES_KEY, WECHAT_TOKEN, WECHAT_APP_ID, KF_ACCOUNT
from crypto import WeChatCrypto
from wechat_api import (
    get_or_upload_media,
    send_image_message,
    send_text_message,
    send_transfer_to_human,
    send_typing_indicator,
)

# ──────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="微信小程序 AI 客服", version="1.0.0")

# MsgId 去重（防微信重试导致消息被处理两次）
_processed_msg_ids: set[str] = set()
_MSG_ID_MAX = 1000

crypto = WeChatCrypto(
    token=WECHAT_TOKEN,
    encoding_aes_key=WECHAT_ENCODING_AES_KEY,
    app_id=WECHAT_APP_ID,
)


# ──────────────────────────────────────────
# GET /webhook — 服务器验证（配置时微信调用一次）
# ──────────────────────────────────────────

@app.get("/webhook")
async def verify_server(
    signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
):
    """
    微信服务器配置验证
    验证成功原样返回 echostr，微信确认服务器有效
    """
    if crypto.verify_get(signature, timestamp, nonce):
        logger.info("服务器验证成功")
        return PlainTextResponse(echostr)
    else:
        logger.warning("服务器验证失败，签名不匹配")
        return PlainTextResponse("forbidden", status_code=403)


# ──────────────────────────────────────────
# POST /webhook — 接收用户消息
# ──────────────────────────────────────────

@app.post("/webhook")
async def receive_message(
    request: Request,
    background_tasks: BackgroundTasks,
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
):
    """
    接收微信推送的客服消息
    处理流程：
    1. 立刻返回 "success"（必须5秒内响应）
    2. BackgroundTasks 在响应发出后继续执行：
       解密 → AI 生成回复 → 调用微信发送 API

    使用 FastAPI BackgroundTasks 而非 asyncio.create_task，
    确保在 SCF 云函数环境中任务能可靠完成。
    """
    body = await request.body()
    body_xml = body.decode("utf-8")

    # 解密并验证消息
    msg = crypto.decrypt_and_parse(body_xml, msg_signature, timestamp, nonce)

    if msg is None:
        logger.warning("消息验签失败，忽略本次请求")
        return PlainTextResponse("success")

    msg_id = msg.get("MsgId", "")
    if msg_id:
        if msg_id in _processed_msg_ids:
            logger.info(f"重复消息忽略 MsgId={msg_id}")
            return PlainTextResponse("success")
        if len(_processed_msg_ids) >= _MSG_ID_MAX:
            _processed_msg_ids.clear()
        _processed_msg_ids.add(msg_id)

    msg_type = msg.get("MsgType", "")
    openid = msg.get("FromUserName", "")

    logger.info(f"收到消息 | openid={openid[:8]}... | type={msg_type}")

    if msg_type == "text":
        user_text = msg.get("Content", "").strip()

        # 转人工：同步返回被动 XML，让微信立刻路由用户进入客服小助手待接入队列
        if needs_human(user_text):
            inner_xml = _build_transfer_xml(openid, timestamp, KF_ACCOUNT)
            reply_xml = crypto.encrypt(inner_xml, timestamp, nonce)
            await send_text_message(
                openid,
                "好的，正在为您转接人工客服，请稍候。\n如暂无客服在线，我们会在工作时间（9:00-18:00）尽快联系您。",
            )
            clear_history(openid)
            logger.info(f"[转人工] 被动回复 openid={openid[:8]}... kf={KF_ACCOUNT or '(auto)'}")
            return Response(content=reply_xml, media_type="application/xml")

        background_tasks.add_task(_handle_text, openid, user_text)

    elif msg_type == "event":
        event = msg.get("Event", "")
        if event == "user_enter_tempsession":
            background_tasks.add_task(_send_welcome, openid)

    else:
        # 图片、语音等暂不支持，友好提示
        background_tasks.add_task(
            send_text_message, openid, "您好，目前仅支持文字消息，请用文字描述您的问题 😊"
        )

    # 必须立刻返回 "success"，否则微信会重试3次
    return PlainTextResponse("success")


# ──────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────

def _build_transfer_xml(openid: str, timestamp: str, kf_account: str = "") -> str:
    """构建 transfer_customer_service 被动回复的内层 XML"""
    trans_info = ""
    if kf_account:
        trans_info = f"<TransInfo><KfAccount><![CDATA[{kf_account}]]></KfAccount></TransInfo>"
    return (
        f"<xml>"
        f"<ToUserName><![CDATA[{openid}]]></ToUserName>"
        f"<FromUserName><![CDATA[{WECHAT_APP_ID}]]></FromUserName>"
        f"<CreateTime>{timestamp}</CreateTime>"
        f"<MsgType><![CDATA[transfer_customer_service]]></MsgType>"
        f"{trans_info}"
        f"</xml>"
    )


# ──────────────────────────────────────────
# 异步处理逻辑
# ──────────────────────────────────────────

async def _handle_text(openid: str, text: str) -> None:
    """处理用户文本消息"""

    # 发送"正在输入"提示（让用户知道消息已收到）
    await send_typing_indicator(openid)

    # 调用 AI 生成回复（同时返回知识库命中的图片链接）
    reply, image_urls = await get_ai_reply(openid, text)
    await log_chat(openid, text, reply)   # 写入 COS 聊天日志

    # 1. 先发文字回复
    success = await send_text_message(openid, reply)
    if success:
        logger.info(f"[文字回复成功] openid={openid[:8]}...")
    else:
        logger.error(f"[文字回复失败] openid={openid[:8]}...")
        return

    # 2. 逐张发送图片（最多2张，避免超出消息条数限制）
    for url in image_urls[:2]:
        try:
            media_id = await get_or_upload_media(url)
            if media_id:
                await send_image_message(openid, media_id)
                logger.info(f"[图片发送成功] openid={openid[:8]}...")
        except Exception as e:
            logger.error(f"[图片发送失败] {url} | {e}")


async def _send_welcome(openid: str) -> None:
    """用户进入客服对话时发送欢迎语"""
    await send_text_message(
        openid,
        "您好！我是 AI 智能客服，很高兴为您服务 😊\n请问有什么可以帮助您的？"
    )


# ──────────────────────────────────────────
# 健康检查（用于确认服务正常运行）
# ──────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "微信小程序AI客服"}
