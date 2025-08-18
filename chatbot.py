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

while True:
  user_input = input("user:")
  if user_input == "exit":
    break
  response = model.invoke(user_input)
  print("AI: ", response.content)