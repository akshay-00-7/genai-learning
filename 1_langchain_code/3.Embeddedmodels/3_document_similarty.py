from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = GoogleGenerativeAIEmbeddings( model = "models/gemini-embedding-001" , dimensions = 300)

documents = [
    "LangChain LLMs ke saath kaam karta hai",
    "Google Gemini free embeddings deta hai",
    "Embeddings text ko numbers mein convert karti hain"
]
query = "tell me about langchain "

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding], doc_embedding)

index,score  = sorted(list(enumerate(score)),key= lambda x:x [1])[-1]

print (query)
print(documents [index])