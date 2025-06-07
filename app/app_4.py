import gradio as gr
from ollama import AsyncClient

client = AsyncClient(host='http://10.182.0.249:11434')


async def traduce(texto, opciones):
    """
    Traduce el texto de entrada a los idiomas seleccionados.
    """
    if not opciones:
        return "Por favor, selecciona al menos un idioma para traducir."

    mensajes = [
        {
            'role': 'user',
            'content': f"Traduce el siguiente texto a: {', '.join(opciones)}. Texto: {texto}"
        }
    ]

    response = await client.chat('llama3.2', messages=mensajes)
    return response['message']['content']

demo = gr.Interface(
    fn=traduce,
    inputs=[
        gr.Textbox(
            label="Texto de entrada",
            placeholder="Pon tu texto para traducir aquí...",
            lines=1,
            value="Hola, ¿cómo estás?",
            max_length=50
        ),
        gr.CheckboxGroup(
        choices=["Español", "Inglés", "Francés", "Alemán", "Latín"],
        label="Lenguajes para traducir",
    ),

    ],
    outputs=gr.Textbox(label="Resultado")
)

demo.launch()