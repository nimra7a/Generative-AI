from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_community.document_loaders import TextLoader
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = TextLoader('cricket.txt' , encoding = 'utf-8')
docs = loader.load()

prompt = PromptTemplate(
    template = "Write the summary of the poem. \n {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'poem' : docs[0].page_content})

print('Poem: \n', docs[0].page_content)
print('\n \n' , result)

