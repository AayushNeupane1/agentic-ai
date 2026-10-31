import ollama


def process_response(response):
    return response.upper()


user_message = "What are your gym opening hours?"

# Sending the user question to ollama and getting the response
response = ollama.chat(
    model="qwen2:7b",
    messages=[
        {"role": "user", "content": user_message}
    ]
)

# Extracting the actual response 
llm_response = response["message"]["content"]

# Processing the LLM response
result = process_response(llm_response)

print("User:", user_message)
print("Original LLM response:", llm_response)
print("Processed response:", result)