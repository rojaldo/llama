from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import MarkdownReader

# Configurar modelos locales de Ollama
Settings.llm = Ollama(
    model="gemma3:1b",  # Modelo de lenguaje
    base_url="http://10.182.0.249:11434",
    temperature=0.1  # Más determinista para búsquedas
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://10.182.0.249:11434"
)

# Cargar documentos desde directorio, unicamente los .md

documents = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".md"],
    file_extractor = {
        ".md": MarkdownReader()
    },
    recursive=True
).load_data()  #

# Crear índice con embeddings locales
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model  # Usar embeddings de Ollama
)

# Configurar motor de consultas con modelo local
query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3  # Número de resultados a considerar
)

# Realizar consulta usando el modelo local
respuesta = query_engine.query("dime que es AlphaStar y que empresa la hizo")
print("Respuesta basada en documentos:\n", respuesta.response)