from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):
  summary : Annotated[str, "Give the summary of the given review."] 
  sentiment : Annotated[Literal["pos", "neg"], "Tell the sentiment of the review either its positive, negative or mixed."]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Thardware is great, but the software feels bloated. 
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. 
Hoping for a software update to fix this.""")

print(result)
print("Summary:", result['summary'])
print("Sentiment: ", result['sentiment'])














