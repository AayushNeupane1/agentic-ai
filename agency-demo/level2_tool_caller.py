import json
import ollama

MODEL = "qwen2:7b"


# Define available tools
def check_membership(member_id):
    return f"Membership for {member_id} is active."


def log_workout(member_id, exercise):
    return f"Successfully logged {exercise} for {member_id}."


def calculate_calories(food):
    calorie_data = {
        "banana": 105,
        "apple": 95,
        "egg": 78
    }

    calories = calorie_data.get(food.lower())

    if calories is None:
        return "Food not found in the sample database."

    return f"{food} contains approximately {calories} calories."


# Tool registry
tools = {
    "check_membership": check_membership,
    "log_workout": log_workout,
    "calculate_calories": calculate_calories
}


user_message = input("What would you like me to do? ")

# Ask LLM to choose a tool and arguments
response = ollama.chat(
    model=MODEL,
    format="json",
    messages=[
        {
            "role": "system",
            "content": """
            You are a gym assistant with access to three tools.

            1. check_membership: requires member_id.
            2. log_workout: requires member_id and exercise.
            3. calculate_calories: requires food.

            Return JSON in this format:
            {
                "tool": "tool_name",
                "arguments": {
                    "argument_name": "value"
                }
            }

            Choose only one of the three available tools.
            """
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
)

# Extract LLM decision
decision = json.loads(response["message"]["content"])

tool_name = decision.get("tool")
arguments = decision.get("arguments", {})

print("\nLLM SELECTED TOOL:", tool_name)
print("LLM SELECTED ARGUMENTS:", arguments)

# Validate and execute the selected tool
if tool_name not in tools:
    print("Invalid tool selected.")

else:
    try:
        result = tools[tool_name](**arguments)
        print("\nTOOL RESULT:", result)

    except TypeError:
        print("Invalid or missing tool arguments.")