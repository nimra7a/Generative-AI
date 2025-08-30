from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN

llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt, model, parser)

def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)

})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'Computer Vision'})
final_result = """{} \nword count: {} """.format(result['joke'], result['word_count'])

print(final_result)








