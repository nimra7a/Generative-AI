from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

#schema
schema = [
    ResponseSchema(name='fact 1', description='Fact 1 of the topic'),
    ResponseSchema(name='fact 2', description='Fact 2 of the topic'),
    ResponseSchema(name='fact 3', description='Fact 3 of the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 Facts of the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#traditional Way
# prompt = template.invoke({'topic' : 'Black Hole'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

#chain
chain = template | model | parser
result = chain.invoke({'topic':'Black Hole'})

print(result)


