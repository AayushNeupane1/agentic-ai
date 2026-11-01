## When we say agency is a spectrum, it means that the capacity to make choices, take action, and exert control over your life is not an "all-or-nothing" trait. You do not either completely have it or completely lack it; instead, it exists in degrees that shift constantly based on your environment, internal state, and external constraints.

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
check_membership()log_workout()calculate_calories()

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
