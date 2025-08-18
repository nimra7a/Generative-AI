from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation",
  max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage(content="You're a helpful Assitant")]

while True:
  user_input = input("user:")
  if user_input == "exit":
    break
  chat_history.append(HumanMessage(content=user_input))
  response = model.invoke(chat_history)
  chat_history.append(AIMessage(content=response.content))
  print("AI: ", response.content)

print(chat_history)