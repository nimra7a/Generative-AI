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

prompt1 = PromptTemplate(
    template = "Generate a detailed report about the {topic}.",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a five line summary of given text \n {text}.",
    input_variables=['text']
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Uneployment in Pakistan'})
print(result)
chain.get_graph().print_ascii()
