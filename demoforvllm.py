from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage, AssistantMessage, RawAudio
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
bcn_file = hf_hub_download("patrickvonplaten/audio_samples", "bcn_weather.mp3", repo_type="dataset")

def file_to_chunk(file: str) -> AudioChunk:
    audio = Audio.from_file(file, strict=False)
    return AudioChunk.from_audio(audio)

text_chunk = TextChunk(text="Which speaker is more inspiring? Why? How are they different from each other?")
user_msg = UserMessage(content=[file_to_chunk(obama_file), file_to_chunk(bcn_file), text_chunk]).to_openai()

print(30 * "=" + "USER 1" + 30 * "=")
print(text_chunk.text)
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=[user_msg],
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content

print(30 * "=" + "BOT 1" + 30 * "=")
print(content)
print("\n\n")
# The speaker who is more inspiring is the one who delivered the farewell address, as they express
# gratitude, optimism, and a strong commitment to the nation and its citizens. They emphasize the importance of
# self-government and active citizenship, encouraging everyone to participate in the democratic process. In contrast,
# the other speaker provides a factual update on the weather in Barcelona, which is less inspiring as it
# lacks the emotional and motivational content of the farewell address.

# **Differences:**
# - The farewell address speaker focuses on the values and responsibilities of citizenship, encouraging active participation in democracy.
# - The weather update speaker provides factual information about the temperature in Barcelona, without any emotional or motivational content.


messages = [
    user_msg,
    AssistantMessage(content=content).to_openai(),
    UserMessage(content="Ok, now please summarize the content of the first audio.").to_openai()
]
print(30 * "=" + "USER 2" + 30 * "=")
print(messages[-1]["content"])
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content
print(30 * "=" + "BOT 2" + 30 * "=")
print(content)