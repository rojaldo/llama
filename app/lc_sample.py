from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# 1. Modelo de validación
class RespuestaTecnica(BaseModel):
    concepto: str
    aplicaciones: list[str]
    complejidad: int

# 2. Pipeline de procesamiento
loader = WebBaseLoader(["https://docs.python.org/3/whatsnew/3.13.html"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

llm = OllamaLLM(model="llama3.2")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()
)

# 3. Ejecución sin validación estructurada
resultado = qa_chain.invoke("What’s New In Python 3.13?")
print(resultado["result"])