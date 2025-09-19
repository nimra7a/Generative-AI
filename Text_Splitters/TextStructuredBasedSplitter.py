from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap= 0
)

loader = PyMuPDFLoader(
    'assignmment.pdf'
)

docs = loader.load()
result = splitter.split_documents(docs)
print(result[0].page_content)