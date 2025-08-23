from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
import os
import secrets

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="openai/gpt-oss-120b",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#schema
class Review(BaseModel):
  key_themes : list[str] = Field(description="Give all the key themes of the review")
  summary : str = Field(description="Give the summary of the given review.")
  sentiment: Literal["pos", "neg"] = Field(description="Tell the sentiment of the review either its positive, negative or mixed.")
  pros: Optional[list[str]] = Field(default = None, description="Write down all the pros inside the list.")
  cons: Optional[list[str]] = Field(default = None, description="Write down all the cons inside the list.")
  name :Optional[list[str]] = Field(default = None,description= "Give the name of the reviewer.")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Review by Nimra Ansari""")

print(result)
print("Key Themes: ", result.key_themes)
print("Summary:", result.summary)
print("Sentiment: ", result.sentiment)
print("Pros: ", result.pros)
print("Cons: ", result.cons)
print("Reviewer: ", result.name)














 