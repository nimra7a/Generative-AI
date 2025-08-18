
# **Sementic Search**

Install the **Dependcies**
"""

!pip install -qU langchain-huggingface

!pip install scikit-learn

"""**Import Packages**"""

from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

"""**Hugging Face Model**"""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

"""**Document**"""

doc = ["1. Cricket has produced many legendary figures over the decades, each leaving their unique mark on the sport. Sachin Tendulkar, known as the “Little Master” from India, is celebrated for his unmatched batting records, including 100 international centuries. His career spanned over two decades, inspiring millions of fans with his dedication, humility, and technical brilliance.",
       "2. In modern cricket, Babar Azam of Pakistan has emerged as one of the finest batsmen across all formats. Known for his elegant stroke play and remarkable consistency, Babar has quickly risen through the ranks to become the captain of the national team. His ability to adapt to different match situations makes him a key player for Pakistan in both domestic and international tournaments.",
       "3. Meanwhile, AB de Villiers from South Africa has been regarded as one of the most versatile and innovative cricketers of his generation. Nicknamed “Mr. 360” for his ability to hit the ball to all parts of the ground, de Villiers redefined aggressive batting. Even after his retirement from international cricket, his performances in various T20 leagues continue to excite cricket enthusiasts worldwide."]

"""**Query**"""

query = "Who is known as Mr. 360 in cricket?"

"""**Embedding Generation**"""

doc_embedding = embeddings.embed_documents(doc)
query_embedding = embeddings.embed_query(query)

"""**Similarity Score**"""

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

"""**Output**"""

print(query)

print(doc[index])

print("similarity" , scores)