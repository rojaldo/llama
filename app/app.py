
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from llama_index.core import SimpleDirectoryReader

from llama_index.readers.structured_data.base import StructuredDataReader

# 1. Configuración de modelos locales
Settings.llm = Ollama(
    model="llama3.2",  # Descargar previamente: ollama pull llama3.2
    temperature=0.3,
    context_window=4096,
    request_timeout=240
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# 2. Cargar e indexar documentos
parser = StructuredDataReader(col_index=[1, -1], col_metadata="col3")
file_extractor = {
    ".xlsx": parser,
    # ".csv": parser,
    # ".json": parser,
    # ".jsonl": parser,
}
documentos = SimpleDirectoryReader(
    "data", file_extractor=file_extractor
).load_data()

documents = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor = {
        ".xlsx": parser,
        # Puedes añadir más extractores si es necesario
    },
    required_exts=[".xls", ".xlsx"],
    recursive=True
).load_data()  #

indice = VectorStoreIndex.from_documents(
    documentos,
    show_progress=True
)

# 3. Crear QueryEngine y QueryEngineTool
query_engine = indice.as_query_engine(llm=Settings.llm)
herramienta_query_engine = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="consulta_documentos",
        description="Herramienta para leer un xlsx y responder preguntas sobre su contenido. " +
                    "Utiliza un modelo de lenguaje para interpretar los datos y generar respuestas.",
        return_direct=True  # Para que la respuesta sea directa sin formato adicional
    )
)

# 4. Crear agente y consultar
agente = ReActAgent.from_tools(
    tools=[herramienta_query_engine],
    llm=Settings.llm,
    verbose=True,
    max_iterations=50  # Increase from default (usually 10)

)

if __name__ == "__main__":
    pregunta = "dame toda la información que tienes del río Tajo"
    respuesta = agente.query(pregunta)
    print("Respuesta:\n", respuesta)