from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.output_parsers import pydantic
from pydantic import BaseModel, Field
from typing import Literal
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback.")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate( 
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables= {'format_instructions' : parser2.get_format_instructions()}

)

classifer_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)



# brach_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )

brach_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative' , prompt3 | model | parser1),
    RunnableLambda(lambda x: "Couldn't find any sentiment!")
)

chain = classifer_chain | brach_chain
result = chain.invoke({'feedback': 'This is a terrible smartphone.'})
print(result)
chain.get_graph().print_ascii()




