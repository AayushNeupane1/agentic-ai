import json
import ollama

MODEL = "qwen2:7b"


# Specialized agent 1
def workout_agent(user_request):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a workout analysis agent.
                Analyze the user's workout-related request.
                Give a short, useful result.
                """
            },
            {
                "role": "user",
                "content": user_request
            }
        ]
    )

    return response["message"]["content"]


# Specialized agent 2
def nutrition_agent(user_request):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a nutrition analysis agent.
                Analyze the user's nutrition-related request.
                Give a short, useful result.
                """
            },
            {
                "role": "user",
                "content": user_request
            }
        ]
    )

    return response["message"]["content"]


# Main coordinator
def coordinator_agent(user_request):

    response = ollama.chat(
        model=MODEL,
        format="json",
        messages=[
            {
                "role": "system",
                "content": """
                You are a fitness workflow coordinator.

                Decide which specialized agents are needed.

                Available agents:
                - workout
                - nutrition

                Return JSON:
                {
                    "agents": ["workout", "nutrition"]
                }

                Select only the agents relevant to the user's request.
                """
            },
            {
                "role": "user",
                "content": user_request
            }
        ]
    )

    decision = json.loads(response["message"]["content"])

    selected_agents = decision.get("agents", [])

    results = []

    # Start selected agentic workflows
    for agent in selected_agents:

        if agent == "workout":
            result = workout_agent(user_request)

        elif agent == "nutrition":
            result = nutrition_agent(user_request)

        else:
            continue

        results.append({
            "agent": agent,
            "result": result
        })

    return results


# Main program
user_request = input("Describe your fitness goal: ")

results = coordinator_agent(user_request)

print("\nFINAL MULTI-AGENT REPORT")

for item in results:
    print(f"\n{item['agent'].upper()} AGENT:")
    print(item["result"])