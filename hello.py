import httpx
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8080/v1",
    http_client=httpx.Client(trust_env=False, timeout=60)  # ignore ALL_PROXY/HTTP(S)_PROXY
)

print([m.id for m in client.models.list().data][:3])
r = client.chat.completions.create(
    model="/media/f/T21/weights/Voxtral-Mini-3B-2507",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(r.choices[0].message.content)

# ['/media/f/T21/weights/Voxtral-Mini-3B-2507']
# Hello! How are you doing today? Is there something specific you would like to talk about or do?
