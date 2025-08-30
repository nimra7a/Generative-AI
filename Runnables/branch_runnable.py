from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda, RunnableBranch
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Write a detail description of {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "Generate the summary of the given text. \n {text}",
    input_variables=['text']
)


parser = StrOutputParser()

detail_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(detail_chain, branch_chain)
result = chain.invoke({'topic': 'Israil vs Palestine'})
print(result)


 