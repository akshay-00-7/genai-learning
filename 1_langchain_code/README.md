# 1. LangChain Code — LLMs, Chat Models & Embeddings

> **Before you read this** — if you've ever used ChatGPT and wondered *"how do developers actually build stuff like this?"* — that's exactly what this folder answers. Start here.

---

## Who is this for?

This folder is for anyone who:
- Knows basic Python but has never touched AI/GenAI code before
- Wants to understand how apps like ChatGPT, Perplexity, or NotebookLM actually work under the hood
- Is starting their GenAI journey and wants a clean, structured starting point

**Prerequisites:**
- Basic Python (variables, functions, loops)
- A free API key from Groq (takes 2 minutes → https://console.groq.com)
- That's it. Nothing else.

---

## What will you understand after going through this folder?

By the time you finish reading and running these files, you will know:

- ✅ What an LLM is and how to call one through code
- ✅ The difference between an LLM and a Chat Model
- ✅ What Embeddings are and why every AI app uses them
- ✅ How semantic search works (the core of ChatPDF, NotebookLM, etc.)
- ✅ How to use 3 different AI providers — Groq, HuggingFace, Google Gemini

---

## The Big Picture — What is GenAI Development?

Before jumping into code, let me explain what's actually happening when you use an AI app.

When you type something into ChatGPT, here's what happens behind the scenes:

```
You type a message
       ↓
Your message goes to OpenAI's servers via API
       ↓
A massive AI model (GPT-4) processes your message
       ↓
The response comes back to your screen
```

Now here's the thing — **every AI company has their own model and their own API.** OpenAI has GPT-4. Google has Gemini. Meta has LLaMA. Each one has different code, different function names, different response formats.

So if you learn OpenAI's way today and tomorrow you want to switch to Gemini — you'd have to rewrite everything.

**LangChain solves this.** It's a Python library that gives you one unified way to talk to any AI model. Change one line — switch from Groq to Gemini to HuggingFace. Everything else stays the same.

```
Your Python Code
      ↓
   LangChain          ← one interface for everything
   /   |   \
Groq  Gemini  HuggingFace  ← any model you want
```

That's why this folder starts with LangChain.

---

## Folder Structure

```
1_langchain_code/
│
├── 1.LLMs/
│   └── 1_llm_demo.py              ← START HERE. Your first ever LLM call.
│
├── 2.chatmodels/
│   ├── 1_chatmodels_groq.py       ← LLM vs Chat Model. Messages explained.
│   └── 2_chatmodels_hf_api.py     ← Open source models via HuggingFace
│
├── 3.Embeddedmodels/
│   ├── 2_Embedded_hf.py           ← What are Embeddings? (runs locally)
│   ├── Embedding_openai_quary.py  ← Embeddings via Google Gemini API
│   └── 3_document_similarty.py    ← Semantic Search — the core of AI apps
│
└── requirements.txt               ← all dependencies in one file
```

**Recommended order:** Go top to bottom, folder by folder. Each file builds on the previous one.

---

## Section 1 — LLMs (Large Language Models)

### What is an LLM?

An LLM (Large Language Model) is a massive AI model trained on billions of pages of text from the internet, books, and code. It learned patterns in language so well that it can now generate human-like text, answer questions, write code, summarize documents — basically anything language-related.

Examples of LLMs:
- GPT-4 (OpenAI)
- LLaMA 3.3 (Meta — free and open source)
- Gemini (Google)
- Claude (Anthropic)

These models are so large (billions of parameters) that you can't run them on your laptop. Companies host them on powerful servers and give you an **API** to access them.

**API** = a way for your code to talk to their servers over the internet. You send a request, they send back a response.

---

### 📄 1_llm_demo.py — Your First LLM Call

**What this file does:** Calls LLaMA 3.3 70B (a powerful open source model running on Groq's servers) with a simple question and prints the answer.

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # reads your API key from .env file

llm = ChatGroq(model="llama-3.3-70b-versatile")  # connect to the model
result = llm.invoke("what is the capital of india")  # send a message

print(result.content)  # print the reply
```

**Line by line breakdown:**

| Line | What it does |
|------|-------------|
| `load_dotenv()` | Reads your `.env` file and loads your API key into memory |
| `ChatGroq(model=...)` | Creates a connection to LLaMA 3.3 running on Groq's servers |
| `.invoke("your question")` | Sends your message and waits for the reply |
| `result.content` | The actual text response from the AI |

**Run it:**
```bash
python 1_llm_demo.py
```

**What you'll see:** The AI's answer to "what is the capital of india" printed in your terminal. That's it. That's a real AI responding to your code.

---

## Section 2 — Chat Models

### LLM vs Chat Model — What's the Difference?

This confused me at first. Here's the clearest way I can explain it:

**LLM (Base Model):**
- Takes raw text, gives raw text back
- No concept of "who is talking"
- Like texting someone who has no context

**Chat Model:**
- Understands conversation structure
- Knows the difference between System instructions, Human messages, and AI replies
- Like a real conversation with roles and context

Every chatbot you've ever used — ChatGPT, Claude, Gemini — is a Chat Model. The base LLM is what's underneath, but the chat interface adds structure on top.

LangChain has 3 message types for this:

```
SystemMessage  → Instructions for the AI ("You are a helpful assistant")
HumanMessage   → What the user says ("Tell me about Python")
AIMessage      → What the AI replied (stored for memory)
```

---

### 📄 1_chatmodels_groq.py — Using Chat Models Properly

**What this file does:** Shows how to use SystemMessage and HumanMessage to structure a proper conversation with the model.

```python
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="how human brain are work.?")
]

result = model.invoke(chat_history)
print(result.content)
```

**Key insight:** Notice that instead of passing a plain string to `.invoke()`, we're passing a **list of messages**. The model reads the whole list and understands — okay, I'm a helpful AI assistant (SystemMessage) and the human is asking me this (HumanMessage).

**Real world connection:** Every time you set a "Custom Instruction" in ChatGPT — that's a SystemMessage. The AI behaves differently based on what you put there.

---

### 📄 2_chatmodels_hf_api.py — Open Source Models via HuggingFace

**What this file does:** Uses HuggingFace's free API to run TinyLlama — an open source model — instead of Groq.

**What is HuggingFace?**
Think of it as GitHub but for AI models. Thousands of open source models, free to use. Anyone can upload a model, anyone can use it.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("what is the capital of india")
print(result.content)
```

**Why this matters:** Same `.invoke()`, same structure — completely different model and provider. That's the power of LangChain. You're not locked into one company's model.

---

## Section 3 — Embeddings

### What are Embeddings? (This is Important)

This is probably the most important concept in modern AI applications. Almost every useful AI app uses embeddings.

**The problem:** AI models can't understand text directly. They understand numbers. So how do we convert the *meaning* of text into numbers?

**The solution: Embeddings**

An embedding converts a piece of text into a **list of hundreds of numbers** (called a vector). These numbers capture the *meaning* of the text — not just the words, but what the text is actually saying.

The magic part:
```
"Delhi is the capital of India"     → [0.23, -0.14, 0.87, 0.03, ...]
"What is India's capital city?"     → [0.21, -0.16, 0.85, 0.04, ...]
"I love eating pizza"               → [-0.45, 0.67, -0.23, 0.91, ...]
```

Notice how the first two sentences (which mean similar things) have similar numbers? And the third one (completely different topic) has very different numbers?

That's embeddings. **Similar meaning = similar numbers.**

This is used in:
- ChatPDF — finding relevant parts of a PDF based on your question
- NotebookLM — understanding what section of a document is relevant
- Google Search — understanding what you *mean*, not just what you *typed*
- Every "chat with your documents" app ever built

---

### 📄 2_Embedded_hf.py — Local Embeddings (No API Key Needed)

**What this file does:** Converts a sentence into an embedding vector using a model that runs completely on your machine.

```python
from langchain_huggingface import HuggingFaceEmbeddings

# This model downloads once (~80MB) then runs offline forever
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India"
vector = embedding.embed_query(text)

print(str(vector))  # you'll see hundreds of numbers
```

**What you'll see when you run this:** A huge list of numbers like `[0.023, -0.142, 0.871, ...]` — that's the meaning of "Delhi is the capital of India" converted to math.

**Why local?** No API key, no internet needed after first download, completely private. Great for sensitive data.

---

### 📄 Embedding_openai_quary.py — Embeddings via Google Gemini

**What this file does:** Same concept — convert text to embedding vector — but using Google's Gemini model via API instead of a local model.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
result = embedding.embed_query("delhi is capital of india")

print(str(result))
```

**Local vs API embeddings — when to use which:**

| | Local (HuggingFace) | API (Gemini) |
|--|--------------------|--------------------|
| Cost | Free forever | Free tier available |
| Quality | Good | Better |
| Speed | Depends on your PC | Fast |
| Privacy | 100% private | Data goes to Google |
| Internet | Not needed | Required |

---

### 📄 3_document_similarty.py — Semantic Search (The Core of AI Apps) 🎯

**What this file does:** This is the big one. Given a query and a list of documents, find which document is most relevant to the query — based on meaning, not keywords.

**How it works — step by step:**

```
Step 1: Convert all documents to embedding vectors
Step 2: Convert the query to an embedding vector  
Step 3: Calculate cosine similarity between query and each document
Step 4: Return the document with the highest similarity score
```

**What is cosine similarity?** It measures how "close" two vectors are. Score of 1 = identical meaning. Score of 0 = completely unrelated.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=300)

documents = [
    "LangChain LLMs ke saath kaam karta hai",
    "Google Gemini free embeddings deta hai",
    "Embeddings text ko numbers mein convert karti hain"
]
query = "tell me about langchain"

# Convert everything to vectors
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Find the most similar document
scores = cosine_similarity([query_embedding], doc_embeddings)
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])  # prints the most relevant document
```

**When you run this:** The query "tell me about langchain" correctly returns the first document about LangChain — even though the words don't exactly match. That's semantic search.

**Real world connection:** This exact code — at a much larger scale — is how ChatPDF works. Upload a 100-page PDF, ask a question, it finds the relevant pages using this exact technique.

---

## Setup — Running These Files

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Create Your `.env` File
Create a file named `.env` in this folder (same level as the .py files):
```
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
HUGGINGFACEHUB_ACCESS_TOKEN=your_hf_token_here
```

⚠️ **Never push `.env` to GitHub.** It's in `.gitignore` so it won't be pushed automatically — but always double-check.

### Step 3 — Get Your Free API Keys

| Provider | Where to get it | Cost |
|----------|----------------|------|
| Groq | https://console.groq.com | Free |
| Google Gemini | https://aistudio.google.com | Free tier |
| HuggingFace | https://huggingface.co/settings/tokens | Free |

### Step 4 — Run the Files
```bash
python 1.LLMs/1_llm_demo.py
python 2.chatmodels/1_chatmodels_groq.py
python 2.chatmodels/2_chatmodels_hf_api.py
python 3.Embeddedmodels/2_Embedded_hf.py
python 3.Embeddedmodels/Embedding_openai_quary.py
python 3.Embeddedmodels/3_document_similarty.py
```

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| LangChain | latest | Unified interface for all AI models |
| Groq | - | Free, blazing fast LLaMA 3.3 70B inference |
| Google Gemini | - | High quality embedding model |
| HuggingFace | - | Open source models (API + local) |
| sentence-transformers | - | Local embedding model |
| scikit-learn | - | Cosine similarity calculation |
| python-dotenv | - | Managing API keys securely |

---

## What's Next?

After going through this folder, you understand the building blocks. The next folder `2_langchain_prompt` builds on everything here — you'll use these same concepts to build actual working apps with a real web interface.

---

*Start with `1_llm_demo.py`. Run it. See the AI respond to your code. Everything else makes more sense after that first moment.* 🚀
