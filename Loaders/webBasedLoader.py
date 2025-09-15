from langchain_community.document_loaders import WebBaseLoader
url = 'https://www.ibm.com/think/topics/machine-learning'
loader = WebBaseLoader(url)

docs = loader.load()
print(docs[0].page_content)