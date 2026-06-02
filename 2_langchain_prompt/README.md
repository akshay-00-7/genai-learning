# LangChain Prompt Engineering — Building Real Things 💬

After the previous folder, I could call an AI model and get a response. Cool. But there was a problem — if I just let users type anything directly into the model, the output quality was all over the place. Sometimes great, sometimes completely off.

That's when I realized — **how you talk to an AI matters just as much as which AI you use.** This folder is me figuring out Prompt Engineering. And by the end of it, I had actual working apps with a real UI.

---

## One Thing to Understand First

LLMs are **stateless**. Completely. Every time you call the model, it remembers absolutely nothing from before. No previous messages, no context, nothing.

So when you use ChatGPT and it "remembers" what you said earlier — that's not the model's memory. That's ChatGPT sending your *entire conversation history* to the model with every single message. The model reads it all fresh each time and *appears* to remember.

Once I understood this, building chatbots started making a lot more sense.

---

## Folder Structure

```
2_langchain_prompt/
│
├── message.py                  ← understanding message types properly
├── chatboat.py                 ← first real chatbot with memory
├── chat_prompt_template.py     ← making prompts dynamic and safe
├── prompt_ui.py                ← wrapped it in a web UI
├── dynamic_prompt_style.py     ← full research paper summarizer app
└── requirements.txt
```

---

## What I Built — File by File

### 📄 message.py — Getting the Foundation Right

Before building anything, I needed to properly understand the three message types. I had touched on this in the previous folder but here I went deeper.

- `SystemMessage` — This is the AI's **briefing**. You tell it who it is, how it should behave, what it should and shouldn't do. The user never sees this.
- `HumanMessage` — What the user says.
- `AIMessage` — What the AI replied. You store this so the model can "remember" it next time.

```python
messages = [
    SystemMessage(content="you are my helpful assistant"),
    HumanMessage(content="tell me about langchain")
]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
```

That last line — `messages.append(AIMessage(...))` — is the key. I'm manually saving the AI's reply so I can pass it back next time. That's how memory works.

---

### 📄 chatboat.py — My First Actual Chatbot

This is where theory became reality. I built a chatbot that runs in the terminal and actually holds a conversation.

The `while True` loop keeps it running. Every message — from me and from the AI — gets added to `chat_history`. Next time the model is called, the whole history goes with it. So it feels like the AI remembers.

```python
chat_history = [SystemMessage(content="you are my helpful AI Assistant")]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    
    print("AI:", result.content)
```

Type "exit" to stop. Simple, clean, works perfectly.

The moment I ran this and had an actual back-and-forth conversation with an AI I built myself — that was a big moment.

---

### 📄 chat_prompt_template.py — Controlling What the User Can Change

Here's a real problem with letting users type freely into an AI:

If I'm building a Cricket expert chatbot and the user asks "explain quantum physics" — the model will just answer it. That's not what I want.

`ChatPromptTemplate` lets me define a **template** where I control the structure and the user only fills in specific variables. Think of it like a form — you can only fill in the blanks, you can't change the form itself.

```python
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Dusra'
})
```

`{domain}` and `{topic}` are the variables. I control what goes in there. The structure of the prompt — how it's worded, how it flows — that stays fixed.

This is how every production AI app is built. Nobody lets raw user input go directly to the model.

---

### 📄 prompt_ui.py — Finally, a Real Web Interface

Everything before this ran in the terminal. Which works, but it's not something you'd show someone who doesn't code.

Here I used **Streamlit** — a Python library that turns your script into a web app with literally 3-4 lines of code. No HTML, no CSS, no JavaScript needed.

```python
st.header('Research Tool')
user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
```

Run it with:
```bash
streamlit run prompt_ui.py
```

A browser tab opens automatically. There's a text box, there's a button, you type something and the AI responds. First time I showed this to someone non-technical, they had no idea I built it in 15 minutes.

---

### 📄 dynamic_prompt_style.py — The Most Complete App Here 🏆

This is everything coming together. A full Streamlit app where the user picks:
- Which research paper they want to understand
- What style of explanation they want (Beginner / Technical / Mathematical)
- How long the explanation should be

Based on all three selections, a detailed prompt is **automatically built** and sent to the model. The user can't go off-track because every choice is a dropdown — controlled inputs only.

```python
prompt = f"""
Explain the research paper '{paper_input}'
in a '{style_input}' style
with a '{length_input}' explanation.
"""
result = model.invoke(prompt)
st.write(result.content)
```

The papers available: Attention Is All You Need (the Transformer paper), BERT, GPT-3, and Diffusion Models. These are literally the papers that shaped modern AI.

Run it:
```bash
streamlit run dynamic_prompt_style.py
```

---

## The Learning Path Through This Folder

If you're reading this to learn, go through the files in exactly this order:

```
message.py                ← understand how messages work
      ↓
chatboat.py               ← build a chatbot, see memory in action
      ↓
chat_prompt_template.py   ← control the prompt structure
      ↓
prompt_ui.py              ← get it off the terminal, into a browser
      ↓
dynamic_prompt_style.py   ← put it all together into a real app
```

Each file builds on the previous one. Skip around and things won't make as much sense.

---

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Create a `.env` file** (never push to GitHub!):
```
GROQ_API_KEY=your_key_here
```

Get your free Groq API key → https://console.groq.com

**Run terminal-based files:**
```bash
python chatboat.py
python message.py
```

**Run Streamlit apps:**
```bash
streamlit run prompt_ui.py
streamlit run dynamic_prompt_style.py
```

---

## Tech Stack

| Tool | What I Used It For |
|------|-------------------|
| LangChain Core | Prompt templates and message handling |
| Groq (LLaMA 3.3 70B) | Fast, free LLM — responses in under a second |
| Streamlit | Turning Python scripts into web apps |
| python-dotenv | Keeping API keys out of the code |

---

*The jump from the previous folder to this one felt significant. Before — I was calling an AI. Here — I was building with it. There's a difference.* 🎯
