# demo_responses_audio.py
import base64, httpx
from openai import OpenAI
from huggingface_hub import hf_hub_download
import os
# 清除代理相关环境变量，确保走直连
for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

# 连接到你本地 vLLM 服务（端口按你的实际为准）
client = OpenAI(
    api_key="EMPTY",
    base_url="http://0.0.0.0:8000/v1",
    http_client=httpx.Client(trust_env=False, timeout=60),  # 忽略环境代理，避免代理拦截 localhost
)

# 你服务里显示出来的模型 id（用 client.models.list() 看过就是本地权重路径）
MODEL_ID = "/media/f/T21/weights/Voxtral-Mini-3B-2507"

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

# 准备两段示例音频（也可以换成你自己的本地文件路径）
bcn   =  '/home/f/.cache/huggingface/hub/datasets--patrickvonplaten--audio_samples/snapshots/9d8cce5666aa185bbe2d72361600b87954485009/bcn_weather.mp3'
obama ='/home/f/.cache/huggingface/hub/datasets--patrickvonplaten--audio_samples/snapshots/9d8cce5666aa185bbe2d72361600b87954485009/obama.mp3'

# --- 第一轮：两段音频 + 提问 ---
messages_round1 = [{
    "role": "user",
    "content": [
        {"type": "input_audio", "audio": {"data": b64(obama), "format": "mp3"}},
        {"type": "input_audio", "audio": {"data": b64(bcn),   "format": "mp3"}},
        {"type": "input_text",  "text": "Which speaker is more inspiring? Why? How are they different?"},
    ],
}]

resp1 = client.responses.create(
    model=MODEL_ID,
    input=messages_round1,
    temperature=0.2,
    top_p=0.95,
)

# 输出文本（不同版本 SDK 可能没有 .output_text，做个兼容）
answer1 = getattr(resp1, "output_text", None)
if not answer1:
    # 简单兜底：把整个对象转成字符串看看
    answer1 = str(resp1)
print("=== BOT 1 ===")
print(answer1)

# --- 第二轮：基于上一轮的回答继续追问（同一“线程”）---
messages_round2 = [
    messages_round1[0],                                     # 第一轮用户消息
    {"role": "assistant", "content": [{"type": "output_text", "text": answer1}]},  # 模型上一轮回复
    {"role": "user", "content": [{"type": "input_text", "text": "Please summarize the first audio."}]},
]

resp2 = client.responses.create(
    model=MODEL_ID,
    input=messages_round2,
    temperature=0.2,
    top_p=0.95,
)

answer2 = getattr(resp2, "output_text", None) or str(resp2)
print("\n=== BOT 2 ===")
print(answer2)
