## What "agency" means here

In this course, **agency is not about a person's control over their life** — it is a technical idea: *how much control the LLM has over the program's flow.* At one end, the model's output is just text that gets printed. At the other, the model decides which branch runs, which function is called with which arguments, whether to loop again, and even whether to start another agent. "Agency is a spectrum" means that control over execution is handed to the model in increasing degrees across the five levels below.

### The five levels of agency


#### ☆☆☆ Simple processor — No agency over program flow
            def process_llm_output(llm_response):    
                return llm_response.upper()

- The LLM generates an answer.
- Python processes that answer.
- The LLM cannot decide what the program does next.

Example: A chatbot that simply answers a question.


#### ★☆☆ Router — Controls basic decisions

            decision = llm_decision()

            if decision == "weather":    
                get_weather()
            else:   
                 search_web()

- The LLM decides which path the program should take.
- Python executes the selected path.
- The LLM doesn't choose arbitrary functions or manage a sequence of actions.
Example: A customer-support bot that routes billing questions to billing support and technical questions to technical support.

#### ★★☆ Tool caller — Chooses which function to execute
            tool_name = llm_choose_tool()
            arguments = llm_choose_arguments()
            
            run_function(tool_name, arguments)

- The LLM chooses a tool.
- It also provides the arguments.
- Python executes the selected function.
Example: A fitness assistant choosing between:
- check_membership()
- log_workout()
- calculate_calories()

The important difference from a router is that the LLM chooses a function to execute, rather than simply selecting between predefined program paths.

#### ★★★ Multi-step agent — Controls iteration and continuation

            while llm_should_continue(): 
               action = llm_choose_next_action()    
               execute_action(action)

- The LLM can perform multiple actions.
- It can evaluate the result of an action.
- It can decide whether another step is necessary.
- It can eventually decide to stop.
Example: An agent that researches a topic, searches for missing information, compares findings, and then produces a final report.

#### ★★★ Multi-agent — Starts another agentic workflow
            if llm_trigger(): 
               execute_another_agent()

- One agentic workflow can initiate another.
- Different agents can handle different responsibilities.
- The workflows can be coordinated to complete a larger task.
Example:
- Research agent gathers information.
- Writing agent drafts a report.
- Review agent checks the report.

### Note: raw Ollama vs smolagents

These five scripts call the model directly with the **raw `ollama` package** (`ollama.chat(...)`) so you can see the mechanics of each level by hand. The Hugging Face course proper builds agents with **smolagents** (`ToolCallingAgent`, `CodeAgent`), which automate levels 2-4 for you — tool-calling, the multi-step loop, and multi-agent coordination. Build these by hand first to understand *what smolagents does under the hood*, then switch to smolagents for real agents.

### Important things to understand

1. Why are we using Ollama?

Ollama provides access to your locally running Qwen2:7b model. All five programs use the same model, so you can focus on the differences in program architecture.

2. Why are we using normal Python functions?

Because an agent needs tools to interact with the outside world. Python functions represent those tools in our demonstrations.

3. Are these production-ready agents?

No. These are educational prototypes. They use sample data, simple prompts, and basic error handling. A real system would need stronger input validation, permissions, logging, retries, and evaluation.

4. What if you get a JSON decoding error?

Qwen may occasionally return invalid JSON. Check the raw response before parsing it:
print(response["message"]["content"])
