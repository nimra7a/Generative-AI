from langchain_core.prompts import ChatPromptTemplate
import os

chat_template = ChatPromptTemplate([
  ('system', "you're a helpful {domain} expert"),
  ('human','Explain simply what is {topic}' )
])

prompt = chat_template.invoke({
  'domain': 'cricket',
  'topic' : 'wicket'
  })

print(prompt)