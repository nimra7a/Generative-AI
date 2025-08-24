from langchain_core.output_parsers.json import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
import secrets

parser = JsonOutputParser()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#traditional way
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

#chaining
chain = template | model | parser 
result = chain.invoke({})
print(result)





