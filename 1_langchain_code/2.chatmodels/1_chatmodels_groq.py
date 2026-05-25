# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGroq(
#     model="llama-3.3-70b-versatile"
# )


# result = model.invoke("what is the capital of india")

# print(result.content)

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="how human brain are work.?")
]

result = model.invoke(chat_history)

print(result.content)


