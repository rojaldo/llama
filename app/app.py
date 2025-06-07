from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configura el modelo Ollama para LLM y embeddings
ollama_llm = Ollama(
    model="gemma3:1b",  # Modelo de lenguaje
    base_url="http://10.182.0.249:11434",
    temperature=0.1  # Más determinista para búsquedas
)

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://10.182.0.249:11434"
)

# Prepara tus documentos (puedes cargar desde archivos, aquí un ejemplo simple)
documents = [
    Document(text="Este es el manual de usuario. Explica las políticas de devolución y garantías."),
    Document(text="Para devolver un producto, contacte con soporte y siga las instrucciones del sitio web.")
]

# Crea el extractor de títulos usando Ollama como LLM
title_extractor = TitleExtractor(llm=ollama_llm, 
                                 title_prompt="Extrae el título de este texto, unicamente el titulo, no des explicaciones de criterios, una frase concisa",
                                 max_length=1)  # Longitud máxima del título

# Define la pipeline de ingesta
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=64, chunk_overlap=0),
        title_extractor,
        embed_model
    ]
)

# Ejecuta la pipeline sobre los documentos
nodes = pipeline.run(documents=documents)

# Visualiza los nodos resultantes
for node in nodes:
    print("--- Nodo ---")
    print("Texto:", node.text)
    print("Título:", node.metadata)
    print("Embeddings (primeros valores):", node.embedding[:5], "...")