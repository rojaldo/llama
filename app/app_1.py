import gradio as gr
import asyncio
from ollama import AsyncClient

# connect to ollama server: localhost:11434
async def main():
    client = AsyncClient()
    messages = [
    {
      'role': 'assistant',
      'content': 'Hello! How can I assist you today?',
    },
  ]

    async def chat(texto):
        message = {
            'role': 'user',
            'content': texto,
        }
        messages.append(message)
        response = await client.chat('llama3.2', messages=messages)
        response_message = {
            'role': 'assistant',
            'content': response['message']['content'],
        } 
        messages.append(response_message)
        return response_message['content']

    demo = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(
            label="Introduce tu texto",
            placeholder="Escribe algo aquí...",
            lines=1,               # Cambia a lines=5 para multilinea
            max_length=100
        ),
        outputs=gr.Textbox(label="Texto invertido")
    )

    demo.launch()

if __name__ == '__main__':
    asyncio.run(main())
