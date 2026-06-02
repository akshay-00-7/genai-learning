from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="City of the person")

# Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Prompt
template = PromptTemplate(
    template="""
    Generate details of a fictional person.

    {format_instructions}
    """,
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Chain
chain = template | model | parser

# Invoke
result = chain.invoke({})

print(result)
print(result.name)
print(result.age)
print(result.city)
print(type(result))