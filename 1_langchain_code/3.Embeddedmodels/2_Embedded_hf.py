from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Local model - No API key needed!
# Pehli baar ~80MB download hoga, phir offline kaam karega
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)

print(str(vector))