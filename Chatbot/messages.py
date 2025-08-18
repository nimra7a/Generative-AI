from langchain_core import messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation",
  max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

messages = [
  SystemMessage(content="You're a helpful Assitant"),
  HumanMessage(content="What is Hugging face?")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)