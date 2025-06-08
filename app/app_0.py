from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import os

# Configurar modelo local
Settings.llm = Ollama(
    model="llama3.2",  # Nombre del modelo en Ollama
    temperature=0.7,
    context_window=8192  # Aumentar ventana de contexto
)

# 1. Cargar documento
documents = SimpleDirectoryReader(
    input_files=["data/ia.md"]
).load_data()

# 2. Dividir en nodos
node_parser = SentenceSplitter(chunk_size=1024)  # Chunks más grandes
nodes = node_parser.get_nodes_from_documents(documents)

# 3. Generar preguntas
preguntas = []
for node in nodes[:3]:
    prompt = f"""
    Genera 2 preguntas relevantes en español basadas en este contenido:
    {node.text}
    
    Las preguntas deben ser:
    - Específicas y concretas
    - Centradas en conceptos clave
    - Formuladas en español neutro
    """
    
    response = Settings.llm.complete(prompt)
    generated = response.text.split("\n")
    preguntas.extend([q.strip() for q in generated if q.strip()])

# 4. Filtrar y mostrar resultados
print("Preguntas generadas:")
for i, pregunta in enumerate(preguntas[:5], 1):
    if pregunta.startswith(("1.", "2.", "- ")):
        pregunta = pregunta[3:].strip()
    print(f"{i}. {pregunta}")
