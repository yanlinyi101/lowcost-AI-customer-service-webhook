# 微信小程序 AI 智能客服 V1.0

基于腾讯云函数（SCF）**python v3.11** 的微信小程序客服 Webhook 服务。用户在小程序客服对话中发送消息，服务自动检索知识库、调用 AI 生成回复，**月费用约 ¥0**。

---

## 功能特性

- **AI 自动回复** — 接入 DeepSeek / OpenAI / Claude 等兼容 OpenAI 格式的模型，开箱即用
- **RAG 知识库检索** — 基于关键词与汉字重叠的双重评分，优先从知识库中检索答案注入提示词
- **多轮对话记忆** — 每个用户独立维护最近 5 轮对话历史
- **转人工触发** — 检测"转人工"等关键词，自动回复转接提示
- **图片回复** — 知识库条目可绑定图片链接，命中时自动随文字一并发送
- **正在输入提示** — AI 处理期间向用户展示"客服正在输入"状态
- **消息安全加密** — 全程使用微信安全模式（AES-256-CBC），防止消息伪造
- **零运维部署** — 运行在腾讯云函数，无需服务器，在免费额度内长期免费

---

## 架构

```
微信用户
  │  发消息
  ▼
微信服务器
  │  POST /webhook（AES 加密 XML）
  ▼
腾讯云函数 SCF（FastAPI）
  ├─ crypto.py      解密 & 验签
  ├─ rag_service.py 检索知识库，提取相关 Q&A
  ├─ ai_service.py  拼装提示词，调用 AI API
  └─ wechat_api.py  调用微信客服消息 API 发送回复
```

---

## 项目结构

```
wechat_ai_service/
├── main.py              FastAPI 入口，Webhook GET/POST 路由
├── ai_service.py        AI 调用、对话历史管理
├── rag_service.py       知识库加载与检索
├── wechat_api.py        微信 API（发消息、上传素材、access_token）
├── crypto.py            微信消息加解密（安全模式）
├── config.py            所有配置项（从环境变量读取）
├── knowledge_base.json  知识库数据文件（可直接编辑）
├── kb_tool.py           知识库管理命令行工具
├── scf_bootstrap        云函数启动脚本
├── requirements.txt     Python 依赖
└── 腾讯云部署说明.md     完整部署教程
```

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <this-repo>
cd wechat_ai_service
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并填写以下变量（本地调试用 `.env` 文件，云函数部署在控制台配置）：

```env
# 微信小程序（mp.weixin.qq.com → 开发设置）
WECHAT_TOKEN=自定义字符串
WECHAT_APP_ID=小程序AppID
WECHAT_APP_SECRET=小程序AppSecret
WECHAT_ENCODING_AES_KEY=43位随机字符串

# AI 服务（默认 DeepSeek，可切换任意兼容 OpenAI 格式的服务商）
AI_API_KEY=your_api_key
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=deepseek-chat
```

### 3. 本地启动

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/health` 返回 `{"status":"ok"}` 即表示服务正常。

### 4. 部署到腾讯云

参见 [腾讯云部署说明.md](./腾讯云部署说明.md)，涵盖打包、创建函数、配置环境变量、开启函数 URL、微信后台配置等完整步骤。

---

## 知识库编辑

知识库文件为 `knowledge_base.json`，每条记录格式如下：

```json
{
  "question": "如何申请退款？",
  "answer":   "支持收到商品后7天无理由退款，请在小程序「我的订单」中申请。",
  "keywords": ["退款", "退货", "不想要", "七天", "7天", "无理由"],
  "image_url": ""
}
```

| 字段 | 说明 |
|------|------|
| `question` | 问题描述，同时参与字符重叠评分 |
| `answer` | 标准答案，原文注入 AI 提示词 |
| `keywords` | 触发词列表，用户消息命中任意一词即匹配（每词 +2 分）；**宁多勿少，多写口语词和同义词** |
| `image_url` | 命中时附带发送的图片链接，无图留空 `""` |

**检索评分规则：**得分 = 关键词命中数 × 2 + 与 question 共同汉字数 × 0.5，高于阈值（默认 1.0）才会注入 AI 提示词，最多注入 3 条。

### 知识库管理工具

```bash
python kb_tool.py list            # 查看所有条目
python kb_tool.py add             # 交互式添加条目
python kb_tool.py delete <序号>   # 删除指定条目
python kb_tool.py export          # 导出为 Excel
python kb_tool.py import          # 从 Excel 导入
```

---

## 切换 AI 服务商

只需在云函数控制台修改环境变量，无需重新上传代码，**修改后立即生效**：

| 服务商 | `AI_BASE_URL` | `AI_MODEL` |
|--------|--------------|------------|
| DeepSeek（推荐） | `https://api.deepseek.com` | `deepseek-chat` |
| OpenAI | `https://api.openai.com` | `gpt-4o-mini` |
| Claude | `https://api.anthropic.com/v1` | `claude-haiku-4-5-20251001` |

---

## 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | FastAPI + uvicorn |
| HTTP 客户端 | httpx（异步） |
| 消息加解密 | pycryptodome（AES-256-CBC） |
| 部署平台 | 腾讯云函数 SCF Web 函数 |
| AI 接口 | OpenAI 兼容格式（默认 DeepSeek） |

---

## License

MIT
