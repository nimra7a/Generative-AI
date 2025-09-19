from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap= 0,
    separator = ''
)

loader = PyMuPDFLoader(
    'assignmment.pdf'
)

docs = loader.load()
result = splitter.split_documents(docs)
print(result[0].page_content)