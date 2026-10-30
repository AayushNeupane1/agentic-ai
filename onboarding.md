# Stage 00 — Onboarding: Running Models Locally with Ollama

> Hugging Face **"Agents"** course — personal walkthrough, documented stage by stage.

**Status: ✅ Complete.** `qwen2:7b` pulled via Ollama, and `smolagents[litellm]` installed in the project virtual environment. The course can now run locally, for free.

---

## Why this stage exists

The Hugging Face Agents course normally calls models through Hugging Face's **online Inference API**. That is the fastest way to start, *but* it uses a free credit allowance. Once you hit the credit limit, your code stops working and you are blocked from practising.

This stage sets up a way to run an AI model on **your own computer** instead of on Hugging Face's servers.

**Benefits**

- No credit limits — experiment as much as you want.
- No internet needed once the model is downloaded.
- Your data stays on your machine (nothing is sent to a company).

**Trade-off:** a local model is smaller and slower than the big hosted ones, and it uses your computer's memory. For learning, that is a perfectly good deal.

### What we achieve by the end of this stage

A model (`qwen2:7b`) running locally, and a Python script that talks to it through smolagents. If this works, **every later stage of the course can run for free** on your machine.

---

## Key terms (learn these before starting)

### What is an "AI agent"?

A program that uses an AI model (an LLM) to **decide what to do**, not just to chat. It can call tools (search the web, run code, read a file), look at the result, and decide the next step. The whole course is about building these. 

AI agents are computer programs or systems designed to make decisions and take actions in a way that appears to be "intelligent." They can be thought of as digital creatures with the ability to perceive their environment, learn from experience, and make decisions based on that learning. Here's a simple breakdown:

1. **Perception**: AI agents receive input about their environment through sensors or observations. This could be as simple as a webcam feeding video to a machine learning model or as complex as interpreting text messages or user interactions.

2. **Reasoning**: Based on their perception, AI agents analyze the data and use logical processes to determine what actions to take next. Thisinvolves algorithms that might use mathematical models, statistical methods, or more complex techniques like neural networks.

3. **Decision Making**: Using their reasoning capabilities, AI agents decide on actions to take. These decisions could be simple, like turninga light on or off, or complex, like choosing the best route in a navigation system or predicting stock market trends.

4. **Learning**: Over time, AI agents can improve their performance by learning from experience. They might adjust their decision-making process based on new data, feedback, or rewards for their actions. This is called machine learning, where algorithms improve their performance as they are exposed to more data.

5. **Actuation**: The final step involves taking physical actions based on the decisions made by the AI agent. For instance, a robot might move its limbs to interact with its environment or a self-driving car might adjust its speed and direction.

AI agents are used in a wide range of applications, from playing video games and answering questions with chatbots to driving cars, analyzing medical images, and even composing music. The key is their ability to adapt and learn, making them increasingly useful in many aspects of humanlife.

### What is smolagents?

A small Python library from Hugging Face for building agents. It handles the boring plumbing: sending your prompt to a model, letting the model call tools, running those tools, and looping until the task is done. You write the logic; smolagents wires it to the model.

### Do we really need smolagents?

Not strictly — you could call a model directly and write the agent loop yourself. But this is the Hugging Face course and it is built around smolagents, so we use it. For a beginner it is the right choice: it removes a lot of low-level code so you can focus on understanding how agents think and act.

### What is Ollama?

A free tool that downloads and runs open LLMs on your own computer. Think of it as a local **engine** that serves a model so your code can send questions to it.

### Why do we need Ollama here?

It is what makes "run the model locally" possible. Without it, there is no model on your machine for smolagents to talk to — you would be back to the credit-limited online API.

### What is LiteLLM / LiteLLMModel?

LiteLLM is a **translator layer**. Many model providers (OpenAI, Ollama, Anthropic, etc.) have slightly different APIs. LiteLLM gives them one common interface. `LiteLLMModel` is the smolagents class that uses LiteLLM, so smolagents can talk to Ollama.

---

## Two separate layers (the thing that confused me)

These are **two different things** and both must be ready:

| Layer | What it is | Where it lives |
|---|---|---|
| **Ollama + qwen2:7b** | the engine and the model | system-level, **outside** the venv |
| **.venv + smolagents/litellm** | the Python tools that *call* the engine | inside the project `.venv` |

Pulling the model with Ollama does **not** put anything in the venv, and installing the Python packages does **not** download a model. Each is done once, separately.

---

## The steps (what to type, and why)

### Step 1 — Install Ollama

Follow the official instructions at <https://ollama.com>.

*Why:* this installs the local engine. Nothing below works until Ollama is on your machine.

### Step 2 — Pull a model locally

```bash
ollama pull qwen2:7b
```

*Why:* `pull` = download the model files to your computer. `qwen2:7b` is a capable, reasonably small open model (the "7b" means 7 billion parameters). The download is a few GB and only happens once. Browse <https://ollama.com> for other models if `qwen2:7b` is too heavy for your machine.

*Check it worked:*

```bash
ollama list
```

You should see `qwen2:7b` listed with its size.

### Step 3 — Start Ollama in the background (one terminal)

```bash
ollama serve
```

*Why:* this starts the local server so your code can send requests to it. It listens at `http://127.0.0.1:11434` (that address means "this computer, port 11434"). Leave this terminal running while you work.

If you see `listen tcp 127.0.0.1:11434: bind: address already in use`, Ollama is **already running** (the installer often starts it for you). That is fine — you do not need to run `ollama serve` again. On Windows, if the install already started Ollama, just skip this step. (The `sudo lsof -i :11434` trick from the course is Mac/Linux only — it does not apply on Windows.)

### Step 4 — Create & activate a virtual environment, then install the package

```powershell
cd "C:\Users\Hp\Desktop\ai agent course"
.venv\Scripts\activate
pip install "smolagents[litellm]"
```

*Why the venv:* a virtual environment is a private box of Python packages just for this project, so it does not clash with other projects. When the prompt shows `(.venv)` at the start, it is active.

*Why `[litellm]`:* this installs smolagents **plus** the extra LiteLLM pieces (the part inside the brackets). Without `[litellm]`, smolagents cannot talk to Ollama.

> Note: `pip install list` is **not** a command. To *see* installed packages, use `pip list`.

### Step 5 — Use LiteLLMModel instead of InferenceClientModel

```python
from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",    # which local model to use
    api_base="http://127.0.0.1:11434",  # where Ollama is listening
    num_ctx=8192,                        # how much text it can hold at once
)
```

What each line means:

- `model_id` — `ollama_chat/` tells LiteLLM to route to Ollama, then the model name you pulled in Step 2.
- `api_base` — the local address of your Ollama server (Step 3).
- `num_ctx` — the **context window**: how many tokens (roughly, words) the model can consider at once. 8192 is a sensible default.

---

## Why this works (the one idea to remember)

- Ollama serves the model locally using an **OpenAI-compatible API**.
- `LiteLLMModel` is built to talk to anything that speaks that API format.
- So you simply swap `InferenceClientModel` → `LiteLLMModel` and change no other code. Everything else in the course stays the same. It is **plug-and-play**: the rest of your agent does not know or care that the model now runs on your laptop instead of in the cloud.

**Mental model:**

```
your agent code  ->  smolagents  ->  LiteLLM  ->  Ollama  ->  model
```

(The online version replaces the last two stages with Hugging Face's API.)

---

## Test for this stage (see `test.py`)

`test.py` creates a `LiteLLMModel` pointing at local Ollama and asks it one question. If you run it and get an answer printed, this stage is done.

```powershell
# Ollama must be running first (Step 3)
python test.py
```

**Common issues**

- `connection refused` → Ollama is not running (do Step 3).
- `model not found` → you did not pull it (Step 2), or the name in `model_id` is misspelled.
- `TypeError: string indices must be integers` → on smolagents 1.26.0, `content` must be a **list of parts** (`[{"type": "text", "text": "..."}]`), not a plain string. See `test.py`.
- `module 'litellm' has no attribute '__version__'` and `pip show ollama` → *not errors.* litellm just doesn't expose `__version__`, and we never install an `ollama` pip package — Ollama is a separate app, reached over its local server.
