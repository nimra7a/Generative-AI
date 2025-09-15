from langchain_community.document_loaders import CSVLoader
file_path = 'Social_Network_Ads.csv'
loader = CSVLoader(file_path=file_path)

docs = loader.load()
print(len(docs))
print(docs[1])