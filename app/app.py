# Instalar dependencias primero (ejecutar en terminal):
# pip install llama-index-core llama-index-embeddings-huggingface llama-index-llms-ollama

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 1. Configurar modelos locales
Settings.llm = Ollama(
    model="llama3.2",  # Descargar previamente: ollama pull llama3.1
    temperature=0.3,
    context_window=4096,
    request_timeout=120
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # Se descarga automáticamente la primera vez
    device="cpu",  # Usar 'cuda' para GPU NVIDIA o 'mps' para Apple Silicon
)

# 2. Cargar e indexar documentos
documentos = SimpleDirectoryReader(
    input_dir="data",  # Carpeta con archivos .txt, .pdf, etc.
    required_exts=[".pdf", ".txt", ".md"]
).load_data()

indice = VectorStoreIndex.from_documents(
    documentos,
    show_progress=True
)

# 3. Crear motor de consulta
motor_consulta = indice.as_query_engine()

# 4. Ejemplo de uso
if __name__ == "__main__":
    respuesta = motor_consulta.query("Explica los conceptos clave del documento")
    print("Respuesta:\n", respuesta)
    print("\nFuentes usadas:", respuesta.get_formatted_sources())
