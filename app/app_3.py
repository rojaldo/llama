import asyncio

import ollama
from ollama import ChatResponse


def current_time() -> str:
  from datetime import datetime
  return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

messages = [{'role': 'user', 'content': 'what time is it?'}]

current_time_tool = {
  'type': 'function',
  'function': {
    'name': 'current_time',
    'description': 'Get the current time in the format YYYY-MM-DD HH:MM:SS',
    'parameters': {
      'type': 'object',
      'properties': {},
      'required': [],
    },
  }
}

available_functions = {
  'current_time': current_time,
}

async def send_request(text):
    client = ollama.AsyncClient()
    response: ChatResponse = await client.chat(
    'llama3.2',
    messages= [{'role': 'user', 'content': text}],
    tools=[current_time_tool],
  )

    if response.message.tool_calls:
      # There may be multiple tool calls in the response
      for tool in response.message.tool_calls:
        # Ensure the function is available, and then call it
        if function_to_call := available_functions.get(tool.function.name):
          print('Calling function:', tool.function.name)
          print('Arguments:', tool.function.arguments)
          output = function_to_call(**tool.function.arguments)
          print('Function output:', output)
          messages.append(response.message)
          messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
          #  write text to the output textbox
          return str(output)

        else:
          print('Function', tool.function.name, 'not found')
          return f'Function {tool.function.name} not found'

      # Return the output of the last tool call

    else:
      print('No tool calls returned from model')
      return response.message.content

async def gradio_app():
  import gradio as gr


  with gr.Blocks() as demo:
    gr.Markdown("Click the button to get the current time.")
    time_button = gr.Button("Get Current Time")
    text = gr.Textbox(label="Prompt", value="")
    time_output = gr.Textbox(label="Current Time")

    time_button.click(send_request, inputs=text, outputs=time_output)

  return demo


async def main():
  # launch the Gradio app
  demo = await gradio_app()
  demo.launch()


if __name__ == '__main__':
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print('\nGoodbye!')