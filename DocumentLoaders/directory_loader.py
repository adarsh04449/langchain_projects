from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path = "Documents",
    glob="*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()
print(docs[38].page_content)
print(docs[38].metadata)

