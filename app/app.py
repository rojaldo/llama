import gradio as gr
import asyncio
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# 1. Configurar modelos locales
llm_correccion = Ollama(model="gemma3:4b", base_url="http://10.182.0.249:11434")
llm_simple = Ollama(model="llama3.2", base_url="http://10.182.0.249:11434")
llm_evaluador = Ollama(model="llama3.2", base_url="http://10.182.0.249:11434")

# 2. Definir cadenas individuales
## Generación técnica
prompt_correccion = PromptTemplate(
    input_variables=["texto"],
    template="Correct the following English text, fixing all grammatical, syntactic, and lexical errors. Return only the corrected version of the text, without adding any comments or explanations. Do not change the content or meaning of the original text: {texto}"
)
chain_correccion = LLMChain(llm=llm_correccion, prompt=prompt_correccion, output_key="texto_correccion")

## Simplificación
prompt_errores = PromptTemplate(
    input_variables=["texto","texto_correccion"],
    template='''Make a list of the grammatical, semantic, and spelling errors found in the following text:

Original text: {texto}

Corrected text: {texto_correccion}

Explain each error in simple terms in SPANISH.'''
)
chain_lista_errores = LLMChain(llm=llm_simple, prompt=prompt_errores, output_key="explicacion_simple")

## Evaluación
prompt_eval = PromptTemplate(
    input_variables=["texto", "texto_correccion"],
    template="""
    Evalúa al usuario en base a su texto original y el texto corregido.
    Considera los siguientes criterios:
    1. Corrección gramatical
    2. Corrección semántica
    3. Corrección ortográfica
    4. Claridad y coherencia del mensaje 
    Texto original del usuario: {texto}
    Texto corregido: {texto_correccion}
    Dale una calificacion numérica del 1 al 10, donde 1 es muy malo y 10 es excelente.
    Si comete menos de 5 errores, califícalo con un 10.
    Si comete entre 5 y 10 errores, califícalo con un 8.
    Si comete dentr 10 y 15 errores, califícalo con un 5.
    Si comete más de 15 errores, califícalo con un 2.
    Explica brevemente por qué le diste esa calificación, mencionando los errores más importantes que cometió el usuario y cómo los corrigió.
    """
)
chain_eval = LLMChain(llm=llm_evaluador, prompt=prompt_eval, output_key="evaluacion")

# 3. Crear SequentialChain
pipeline_completo = SequentialChain(
    chains=[chain_correccion, chain_lista_errores, chain_eval],
    input_variables=["texto"],
    output_variables=["texto_correccion", "explicacion_simple", "evaluacion"],
    verbose=True
)

# 4. Ejecutar flujo completo

def process_text(texto):
    # Invertir el texto
    resultado = pipeline_completo({"texto": texto})
    print("Texto corregido:", resultado["texto_correccion"])
    print("Explicación simple de errores:", resultado["explicacion_simple"])
    print("Evaluación del texto:", resultado["evaluacion"])

    return resultado["texto_correccion"], resultado["explicacion_simple"], resultado["evaluacion"]

async def main():

    demo = gr.Interface(
        fn=process_text,
        inputs=gr.Textbox(
            label="Introduce tu texto",
            placeholder="Escribe algo aquí...",
            lines=25,
            value='''Yesterday, I was go to the park with my friend Maria. We was very excited because the weather was sun and the sky was blue. When we arrive to the park, we see many peoples playing footballs and running around the trees. Maria bringed her dog, but the dog was not very happy because he wanted to sleep in the house. I told Maria, "Why you bring the dog if he not like the parks?" She say nothing, just smile and throwed a stick to the dog. The dog runned but he not catch the stick, he just sit down and look at us with sad eyes.

We decide to make a picnic under a big tree. I forgetted to bring the sandwiches, so we only have some apples and a bottle of water. Maria was angry at me, she said, "How you forget the food? Now we hungry and the dog is more sad." I try to make her laugh by telling a joke, but she not laugh, she just look at the sky and sigh hardly.

After some times, it start to rain and all the peoples in the park runned to their homes. We was also wet and cold, so we decide to go home. On the way, Maria losed her keys and we spend a lot of time looking for them. Finally, we finded the keys near the bench where we was sitting before. When we arrive to Maria’s house, the dog runned inside very fast and jump on the sofa. Maria said, "Next time, you bring the food and I bring the good weather." I laugh and say, "Deal!"'''
            
        ),
        outputs=[gr.Textbox(label="Texto Corregido"),
                 gr.Textbox(label="Explicación Simple de Errores"),
                 gr.Textbox(label="Evaluación del Texto")],
    )

    demo.launch()

if __name__ == '__main__':
    asyncio.run(main())
