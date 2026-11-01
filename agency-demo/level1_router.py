import ollama

MODEL = "qwen2:7b"

# Available functions
def membership_info():
    return "Gym membership costs Rs. 2,000 per month."

def opening_hours():
    return "The gym is open from 6 AM to 9 PM."

def gym_location():
    return "Our gym is located in Kathmandu."

def unknown_request():
    return "Sorry, I cannot help with that request."


# Ask the LLM to classify the user's request
user_message = input("What would you like to know? ")

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": """
            You are a gym request classifier.

            Classify the user's request into exactly one category:
            membership, hours, location, unknown.

            Return only the category name.
            """
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
)

decision = response["message"]["content"].strip().lower()

print("\nLLM DECISION:", decision)

# Python follows the LLM's decision.
# Use substring matching: a local 7B model may answer "hours." or
# "Category: hours" instead of the bare word, so == would miss it.
if "membership" in decision:
    result = membership_info()

elif "hour" in decision:
    result = opening_hours()

elif "location" in decision:
    result = gym_location()

else:
    result = unknown_request()

print("\nASSISTANT:", result)