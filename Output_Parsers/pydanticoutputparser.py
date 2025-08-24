from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

#pydantic class
class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt=18 , description="age of the person")
    city : str = Field(description="name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

#template
template = PromptTemplate(
    template = 'Give the name, age and city of the fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#chain
chain = template | model | parser
result = chain.invoke({'place' : 'Pakistan'})
print(result)




