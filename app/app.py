# Instala los paquetes necesarios (si aún no lo has hecho)
# %pip install llama-index llama-index-experimental llama-index-llms-ollama

import pandas as pd
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

def main():
    # Carga el CSV
    df = pd.read_csv("data/titanic.csv")
    
    # Configuración de Ollama
    llm = Ollama(model="llama3.2", base_url="http://localhost:11434")
    
    # Define las instrucciones y prompts
    instruction_str = (
        "1. Convierte la consulta a código Python ejecutable usando Pandas.\n"
        "2. La última línea debe ser una expresión Python evaluable con eval().\n"
        "3. El código debe resolver la consulta.\n"
        "4. SOLO IMPRIMA LA EXPRESIÓN.\n"
        "5. No ponga la expresión entre comillas.\n"
    )

    pandas_prompt_str = (
        "Trabajas con un dataframe de pandas en Python.\n"
        "El nombre del dataframe es `df`.\n"
        "Esto es el resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Sigue estas instrucciones:\n"
        "{instruction_str}\n"
        "Consulta: {query_str}\n\n"
        "Expresión:"
    )

    response_synthesis_prompt_str = (
        "Dada una pregunta de entrada, sintetiza una respuesta a partir de los resultados de la consulta.\n"
        "Consulta: {query_str}\n\n"
        "Instrucciones de Pandas (opcional):\n{pandas_instructions}\n\n"
        "Salida de Pandas: {pandas_output}\n\n"
        "Respuesta: "
    )

    # Crea los prompts
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    # Construye el pipeline de consulta
    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")

    # Ejemplo de consulta
    response = qp.run(
        query_str="¿Cuál es la edad media de los pasajeros supervivientes?"
    )
    
    print("\nRespuesta final:")
    print(response.message.content)

if __name__ == "__main__":
    main()
