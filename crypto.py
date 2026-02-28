"""
微信消息加解密模块
支持安全模式（AES-256-CBC）
参考微信官方加解密方案：
https://developers.weixin.qq.com/miniprogram/dev/framework/server-ability/message-push.html

注意：小程序消息推送外层封包为 XML，解密后内层消息为 JSON 格式。
"""

import base64
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

from Crypto.Cipher import AES


class WeChatCrypto:
    def __init__(self, token: str, encoding_aes_key: str, app_id: str):
        self.token = token
        self.app_id = app_id
        # EncodingAESKey 是44位Base64字符，解码后得到32字节AES密钥
        self.aes_key = base64.b64decode(encoding_aes_key + "=")

    # ──────────────────────────────────────────
    # 签名验证
    # ──────────────────────────────────────────

    def _sha1(self, *args) -> str:
        """对多个字符串排序后拼接做 SHA-1"""
        items = sorted(args)
        return hashlib.sha1("".join(items).encode("utf-8")).hexdigest()

    def verify_get(self, signature: str, timestamp: str, nonce: str) -> bool:
        """验证 GET 请求签名（服务器配置时使用）"""
        return self._sha1(self.token, timestamp, nonce) == signature

    def verify_post(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str) -> bool:
        """验证 POST 请求签名（安全模式）"""
        return self._sha1(self.token, timestamp, nonce, encrypt) == msg_signature

    # ──────────────────────────────────────────
    # 消息解密
    # ──────────────────────────────────────────

    def decrypt(self, encrypt_msg: str) -> str:
        """
        解密微信消息
        解密后明文格式：
            16字节随机串 | 4字节消息长度(网络序) | 消息内容 | AppId
        """
        decoded = base64.b64decode(encrypt_msg)
        # IV 取 AES key 前16字节
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        plain = cipher.decrypt(decoded)

        # 去除 PKCS7 填充
        pad_len = plain[-1]
        plain = plain[:-pad_len]

        # 跳过16字节随机串，读4字节长度
        msg_len = struct.unpack(">I", plain[16:20])[0]
        msg_xml = plain[20: 20 + msg_len].decode("utf-8")
        return msg_xml

    # ──────────────────────────────────────────
    # XML 解析
    # ──────────────────────────────────────────

    @staticmethod
    def parse_xml(xml_str: str) -> dict:
        """将微信消息 XML 解析为字典"""
        root = ET.fromstring(xml_str)
        result = {}
        for child in root:
            result[child.tag] = child.text or ""
        return result

    def decrypt_and_parse(self, body_xml: str, msg_signature: str,
                          timestamp: str, nonce: str) -> dict | None:
        """
        完整处理 POST 消息：验签 → 解密 → 解析
        返回消息字典，验签失败返回 None

        小程序消息推送：外层封包为 XML，解密后内层为 JSON。
        """
        outer = self.parse_xml(body_xml)
        encrypt = outer.get("Encrypt", "")

        if not self.verify_post(msg_signature, timestamp, nonce, encrypt):
            return None

        decrypted_str = self.decrypt(encrypt)

        # 小程序内层消息为 JSON，兼容性地尝试 JSON 优先，失败则退回 XML
        try:
            return json.loads(decrypted_str)
        except (json.JSONDecodeError, ValueError):
            return self.parse_xml(decrypted_str)
