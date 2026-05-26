# LangChain Code — My First Steps into GenAI 🚀

Okay so let me be honest — when I first heard about LangChain, I had no idea what it was. I knew ChatGPT existed, I knew AI was a big thing, but I had zero clue how developers actually *use* these AI models in their own code.

This folder is literally where everything started for me. No fancy apps, no complex pipelines — just me figuring out how to talk to an AI model through Python code for the very first time.

---

## The Problem I Was Trying to Solve

Every AI company — OpenAI, Google, Meta, Groq — has their own way of calling their models. Different functions, different parameters, different response formats. So if I learned OpenAI's way today and tomorrow I want to switch to Gemini, I'd have to rewrite everything.

LangChain fixes this. It's basically a **universal remote** for AI models. Same code structure, any model. That's why I started here.

---

## Folder Structure

```
1_langchain_code/
│
├── 1.LLMs/
│   └── 1_llm_demo.py              ← my very first LLM call
│
├── 2.chatmodels/
│   ├── 1_chatmodels_groq.py       ← learned SystemMessage, HumanMessage
│   └── 2_chatmodels_hf_api.py     ← tried open-source model via HuggingFace
│
├── 3.Embeddedmodels/
│   ├── 2_Embedded_hf.py           ← what are embeddings? (local model)
│   ├── Embedding_openai_quary.py  ← embeddings via Google Gemini API
│   └── 3_document_similarty.py    ← built my first semantic search!
│
└── requirements.txt
```

---

## What I Learned — File by File

### 📄 1.LLMs / 1_llm_demo.py — The "Hello World" of GenAI

This is literally the first GenAI code I ever wrote. 10 lines. That's it.

I used **Groq** (a free API that runs Meta's LLaMA model super fast) and just asked it a simple question. The moment I ran this and saw the AI reply in my terminal — that feeling was different.

```python
llm = ChatGroq(model="llama-3.3-70b-versatile")
result = llm.invoke("what is the capital of india")
print(result.content)
```

One thing I understood here — `invoke()` is how you send a message to the model and get a reply. Simple.

---

### 📄 2.chatmodels / 1_chatmodels_groq.py — Wait, LLM and Chat Model are Different?

Yeah I didn't know this either. Here's the difference:

- **LLM** = raw text in, raw text out. It's the base model, no conversation awareness.
- **Chat Model** = understands *who* is talking. It knows the difference between the System (instructions), the Human (user), and the AI (its own previous replies).

This is where I learned about the 3 message types that every chatbot is built on:

```python
chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),  # AI ki personality
    HumanMessage(content="how human brain are work.?")         # mera sawaal
]
result = model.invoke(chat_history)
```

`SystemMessage` is basically you telling the AI "behave like this." Every ChatGPT custom instruction you've ever set — that's a SystemMessage behind the scenes.

---

### 📄 2.chatmodels / 2_chatmodels_hf_api.py — Free Open Source Models!

So Groq is free but still someone else's model. Here I tried **HuggingFace** — a platform with thousands of open-source AI models that anyone can use.

I used TinyLlama — a small but capable model. Same LangChain interface, completely different provider. That's when I really understood why LangChain exists.

```python
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
```

---

### 📄 3.Embeddedmodels / 2_Embedded_hf.py — What on Earth are Embeddings?

Okay this took me a minute to understand. Let me explain it the way I finally got it.

AI can't understand text directly. It understands **numbers**. So embeddings are basically a way to convert text into a list of numbers (called a **vector**) that captures the *meaning* of the text.

The cool part? Similar sentences will have similar numbers. So "Delhi is the capital of India" and "What is India's capital city?" will have very similar vectors — even though the words are different.

In this file I used a HuggingFace model that runs **completely locally on my machine**. No API key needed. First run downloads ~80MB, after that it works offline.

```python
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector = embedding.embed_query("Delhi is the capital of India")
# vector = [0.023, -0.14, 0.87, ...] hundreds of numbers
```

---

### 📄 3.Embeddedmodels / Embedding_openai_quary.py — Same Thing, Google's Way

Same concept, but using Google's **Gemini Embedding model** via API. The output quality is better than the local model. I compared both and understood the trade-off — local = private + offline, API = better quality but needs internet.

---

### 📄 3.Embeddedmodels / 3_document_similarty.py — My First Semantic Search 🎯

This is the most exciting file in this whole folder. I actually built something useful here.

The idea: give it a query, and it finds the most *relevant* document from a list — not by keyword matching, but by **meaning**.

How it works:
1. Convert all documents into embedding vectors
2. Convert the query into a vector  
3. Calculate **cosine similarity** — basically measures how "close" two vectors are
4. Return the document with the highest similarity score

```python
documents = [
    "LangChain LLMs ke saath kaam karta hai",
    "Google Gemini free embeddings deta hai",
    "Embeddings text ko numbers mein convert karti hain"
]
query = "tell me about langchain"

# compare karo query ko har document se
score = cosine_similarity([query_embedding], doc_embedding)
```

When I ran this — it actually returned the right document! That moment I realized this is exactly how **ChatPDF, NotebookLM, and every "chat with your documents" app** works. Just on a bigger scale.

---

## Setup — How to Run These Files

**Step 1 — Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Create a `.env` file** in this folder (never push this to GitHub!):
```
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
HUGGINGFACEHUB_ACCESS_TOKEN=your_key_here
```

**Step 3 — Run any file:**
```bash
python 1_llm_demo.py
```

**Get free API keys from:**
- Groq → https://console.groq.com (completely free, super fast)
- Google Gemini → https://aistudio.google.com (free tier available)
- HuggingFace → https://huggingface.co/settings/tokens (free)

---

## Tech Stack

| Tool | What I Used It For |
|------|-------------------|
| LangChain | One interface for all AI models |
| Groq | Free + blazing fast LLaMA 3.3 70B inference |
| Google Gemini | High quality embedding model |
| HuggingFace | Open-source models (API + local) |
| sentence-transformers | Local embedding model (no API key) |
| scikit-learn | Cosine similarity calculation |
| python-dotenv | Keeping API keys safe |

---

*If you're starting your GenAI journey and landed here — start with `1_llm_demo.py`. Run it, see the output, feel the dopamine hit, then move forward. That's exactly what I did.* 😄
