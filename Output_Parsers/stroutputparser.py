from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import secrets

from pydantic_core.core_schema import json_schema

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

#1st Prompt
template1 = PromptTemplate(
    template = 'Write detail report on the  {topic}',
    input_variables=['topic']
)


#2nd prompt
template2 = PromptTemplate(
    template = 'Write down the summary of the given text. /n {text}',
    input_variables= '[text]'
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'black hole'})

print(result)
