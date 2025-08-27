from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about the {topic}.",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic' : 'Bedminton'})
print(result)
chain.get_graph().print_ascii()









