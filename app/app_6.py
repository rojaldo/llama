from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings

# Configurar modelos locales de Ollama
Settings.llm = Ollama(
    model="gemma3:4b",  # Modelo de lenguaje
    base_url="http://10.182.0.249:11434",
    temperature=0.1  # Más determinista para búsquedas
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://10.182.0.249:11434"
)

# Documentos de ejemplo sobre errores 404
documents = [
    Document(
        text="En 2016, AlphaGo, desarrollado por DeepMind, derrotó al campeón mundial de Go, Lee Sedol, en una serie de juegos históricos. AlphaGo demostró la capacidad de las redes neuronales para dominar un juego complejo de estrategia, superando el nivel humano.",
        metadata={"model": "AlphaGo", "company": "DeepMind"}
    ),
    Document(
        text="En 2017, DeepMind presentó AlphaZero, un sistema de IA capaz de aprender a jugar Go, ajedrez y shogi sin datos de entrenamiento humanos.",
        metadata={"model": "AlphaZero", "company": "DeepMind"}
    ),
    Document(
        text="En 2019, DeepMind presentó AlphaStar, un sistema de IA capaz de jugar StarCraft II a nivel de los mejores jugadores humanos.",
        metadata={"model": "AlphaStar", "company": "DeepMind"}
    )
]

# Crear índice con configuración local
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model
)

# Configurar motor de chat con el modelo local
query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3,  # Número de resultados a considerar
)

# Consulta usando el modelo local
respuesta = query_engine.query("que tienen en comun AlphaStar y Alphago?")
print("Respuesta basada en documentos:\n", respuesta.response)