from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

st.header('Reasearch Tool')
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

if st.button("Summarize"):

    prompt = f"""
    Explain the research paper '{paper_input}'
    in a '{style_input}' style
    with a '{length_input}' explanation.
    """

    result = model.invoke(prompt)

    st.write(result.content)
