
import os
import json
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable

# 1. Cargar los comentarios desde archivo
BASE_DIR = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "data", "input.json")

with open(json_path, "r", encoding="utf-8") as f:
    datos = json.load(f)

df = pd.DataFrame(datos)

# 2. Configurar el modelo
llm = OllamaLLM(model="gemma3:4b", base_url="http://10.182.0.249:11434")

# 3. Prompt robusto con ejemplos y formato estricto
prompt = PromptTemplate(
    input_variables=["comentario"],
    template=(
        "Eres un analista experto en reseñas de películas. Lee el comentario y "
        "clasifícalo estrictamente como uno de estos tres sentimientos: "
        "'positivo', 'neutro' o 'negativo'.\n\n"
        "Responde únicamente con una palabra.\n\n"
        "Ahora clasifica este comentario:\n\"\"\"\n{comentario}\n\"\"\"\n\n"
        "Sentimiento:"
    )
)

# 4. Crear el pipeline de clasificación
sentiment_classifier: Runnable = prompt | llm

# 5. Función para limpiar y validar la respuesta del modelo
def clasificar_sentimiento(texto):
    resultado = sentiment_classifier.invoke({"comentario": texto})
    sentimiento = resultado.strip().lower()
    if "positivo" in sentimiento:
        return "positivo"
    elif "neutro" in sentimiento:
        return "neutro"
    elif "negativo" in sentimiento:
        return "negativo"
    else:
        return "indeterminado"

# 6. Aplicar clasificación
df["sentimiento_calculado"] = df["comentario"].apply(clasificar_sentimiento)

# 7. Mostrar clasificaciones inválidas (si las hay)
indeterminados = df[df["sentimiento_calculado"] == "indeterminado"]
if not indeterminados.empty:
    print("\n⚠️ Comentarios con clasificación no válida:")
    print(indeterminados[["id", "comentario", "sentimiento_calculado"]])

# 8. Calcular y mostrar porcentajes
conteo = df[df["sentimiento_calculado"] != "indeterminado"]["sentimiento_calculado"] \
            .value_counts(normalize=True) * 100
conteo = conteo.reindex(["positivo", "neutro", "negativo"], fill_value=0).round(2)

print("\n📊 Porcentaje de comentarios por sentimiento:")
print(conteo.to_string())

# 9. (Opcional) Guardar resultados
df.to_csv(os.path.join(BASE_DIR, "comentarios_clasificados.csv"), index=False)
