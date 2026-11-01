import json
import ollama

MODEL = "qwen2:7b"


# Available tools
def check_membership(member_id):
    return f"Membership for {member_id} is active."


def check_attendance(member_id):
    return f"{member_id} attended the gym 12 times this month."


def give_recommendation():
    return "Maintain your workout consistency and follow a balanced diet."


tools = {
    "check_membership": check_membership,
    "check_attendance": check_attendance,
    "give_recommendation": give_recommendation
}


user_request = input("What should I analyze? ")

history = []

# Agent loop
for step in range(5):

    response = ollama.chat(
        model=MODEL,
        format="json",
        messages=[
            {
                "role": "system",
                "content": """
                You are a multi-step gym assistant.

                Available tools:

                1. check_membership: requires member_id.
                2. check_attendance: requires member_id.
                3. give_recommendation: requires no arguments.

                You can call one tool at a time.

                Look at the previous tool results before choosing
                your next action.

                If you have enough information, choose "finish".

                Return JSON in this format:

                {
                    "action": "tool_name_or_finish",
                    "arguments": {
                        "member_id": "M101"
                    }
                }
                """
            },
            {
                "role": "user",
                "content": f"""
                User request: {user_request}

                Previous actions and results:
                {json.dumps(history)}
                """
            }
        ]
    )

    decision = json.loads(response["message"]["content"])

    action = decision.get("action")
    arguments = decision.get("arguments", {})

    print(f"\nSTEP {step + 1}: {action}")

    if action == "finish":
        print("\nAgent decided to finish.")
        break

    if action not in tools:
        print("Unknown action. Stopping agent.")
        break

    try:
        result = tools[action](**arguments)

    except TypeError:
        print("Invalid tool arguments. Stopping agent.")
        break

    print("TOOL RESULT:", result)

    # Save result so the LLM can observe it next time
    history.append({
        "action": action,
        "result": result
    })

else:
    print("\nMaximum steps reached.")