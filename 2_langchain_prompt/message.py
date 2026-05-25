from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

messages =[
    SystemMessage(content="you are my helpful assistent"),
    HumanMessage(content= "tell me about langchain")
]
result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
print(messages)
