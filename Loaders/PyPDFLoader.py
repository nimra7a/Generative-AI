from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader('SoftwareTesting.pdf')
docs = loader.load()

prompt = PromptTemplate(
    template = "Write the summary of the uploaded PDF of Software Testing. \n {pdf}",
    input_variables=['pdf']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'pdf' : docs[0].page_content})

print('\n \n' , result)

