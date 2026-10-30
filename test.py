from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",    # local model pulled via Ollama
    api_base="http://127.0.0.1:11434",  # Ollama's local server
    num_ctx=8192,                        # context window size
)

# smolagents 1.26.0 wants content as a LIST of parts, not a plain string.
messages = [
    {"role": "user", "content": [{"type": "text", "text": "Explain AI agents in simple terms."}]}
]

response = model.generate(messages)
print(response.content)  
