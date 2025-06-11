
import gradio as gr
import requests
import html
import unicodedata
from langchain_community.llms import Ollama

# Variables globales
pregunta_actual = ""
pregunta_traducida = ""
respuesta_correcta = ""
respuesta_correcta_es = ""

# Configura el modelo Ollama
llm_traductor = Ollama(model="gemma3:4b", base_url="http://10.182.0.249:11434")

def normaliza(texto):
    # Quita tildes, pasa a minúsculas y elimina artículos comunes
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    for art in ["el ", "la ", "los ", "las ", "un ", "una ", "unos ", "unas "]:
        if texto.startswith(art):
            texto = texto[len(art):]
    return texto.strip()

# Función para traducir usando el modelo
def traducir_pregunta(pregunta):
    prompt = (
        "Traduce al español de forma natural y clara la siguiente pregunta de trivial. "
        "Devuelve SOLO la pregunta traducida, sin explicaciones ni opciones, y sin comillas:\n\n"
        f"{pregunta}"
    )
    return llm_traductor.invoke(prompt).strip()

def traducir_respuesta(respuesta):
    prompt = (
        "Traduce al español de forma natural y breve la siguiente respuesta de trivial. "
        "Devuelve SOLO la respuesta traducida, sin explicaciones ni comillas:\n\n"
        f"{respuesta}"
    )
    return llm_traductor.invoke(prompt).strip()

# Función para obtener una pregunta
def obtener_pregunta():
    global pregunta_actual, pregunta_traducida, respuesta_correcta, respuesta_correcta_es
    url = "https://opentdb.com/api.php?amount=1&type=multiple"
    response = requests.get(url).json()
    resultado = response["results"][0]
    
    pregunta_actual = html.unescape(resultado["question"])
    respuesta_correcta = html.unescape(resultado["correct_answer"])
    pregunta_traducida = traducir_pregunta(pregunta_actual)
    respuesta_correcta_es = traducir_respuesta(respuesta_correcta)
    
    # Devuelve la pregunta traducida y limpia los otros campos
    return pregunta_traducida, "", ""

def evaluar_respuesta_con_llm(pregunta, respuesta_usuario, respuesta_correcta, respuesta_correcta_es):
    prompt = (
        "Eres un profesor de trivial bilingüe. "
        "Evalúa si la respuesta del usuario es correcta para la siguiente pregunta, "
        "teniendo en cuenta la respuesta oficial en inglés y en español. "
        "Sé flexible: acepta respuestas equivalentes, sin exigir literalidad, y permite respuestas que incluyan solo el dato clave (por ejemplo, el año). "
        "Responde SOLO con ✅ si es correcta o ❌ si es incorrecta, seguido de una breve explicación en español.\n\n"
        f"Pregunta: {pregunta}\n"
        f"Respuesta oficial (inglés): {respuesta_correcta}\n"
        f"Respuesta oficial (español): {respuesta_correcta_es}\n"
        f"Respuesta del usuario: {respuesta_usuario}\n"
        "¿Es correcta la respuesta del usuario?"
    )
    return llm_traductor.invoke(prompt).strip()

# Función para verificar la respuesta del usuario (muy permisiva)
def verificar_respuesta(respuesta_usuario):
    if not pregunta_actual:
        return "Primero haz clic en 'Nueva pregunta'."
    resultado_llm = evaluar_respuesta_con_llm(
        pregunta_actual, respuesta_usuario, respuesta_correcta, respuesta_correcta_es
    )
    return resultado_llm

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Trivia App - Open Trivia DB (con traducción y verificación flexible)")
    
    pregunta_texto = gr.Textbox(label="Pregunta (traducida)", interactive=False)
    entrada_usuario = gr.Textbox(label="Tu respuesta")
    salida = gr.Textbox(label="Resultado", interactive=False)
    
    btn_pregunta = gr.Button("🎲 Nueva pregunta")
    btn_verificar = gr.Button("📩 Verificar respuesta")

    # Cambia aquí: outputs=[pregunta_texto, entrada_usuario, salida]
    btn_pregunta.click(fn=obtener_pregunta, outputs=[pregunta_texto, entrada_usuario, salida])
    btn_verificar.click(fn=verificar_respuesta, inputs=entrada_usuario, outputs=salida)

# Ejecutar la app
demo.launch()
