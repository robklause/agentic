from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3.5:9b-mlx",
    messages=[{"role": "user", "content": "Reply with exactly: local stack is working"}],
)
print(response.choices[0].message.content)