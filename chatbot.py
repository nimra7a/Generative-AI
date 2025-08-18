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
chat_history = []

while True:
  user_input = input("user:")
  if user_input == "exit":
    break
  chat_history.append(user_input)
  response = model.invoke(chat_history)
  chat_history.append(response.content)
  print("AI: ", response.content)

print(chat_history)