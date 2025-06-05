# Inteligencia artificial

## Introducción

La inteligencia artificial (IA) es la inteligencia exhibida por máquinas. En ciencias de la computación, una máquina "inteligente" ideal es un agente racional flexible que percibe su entorno y lleva a cabo acciones que maximicen sus posibilidades de éxito en algún objetivo o tarea. Colloquialmente, el término inteligencia artificial se aplica cuando una máquina imita las funciones "cognitivas" que los humanos asocian con otras mentes humanas, como "aprender" y "resolver problemas".

## Historia

El término "inteligencia artificial" fue acuñado por John McCarthy en 1956, en la conferencia de Dartmouth College, Massachusetts. La inteligencia artificial es un campo de estudio multidisciplinario que busca, mediante el uso de modelos computacionales, el desarrollo de sistemas que puedan realizar tareas que, de momento, requieren inteligencia humana para ser realizadas. La inteligencia artificial es una rama de la informática que se ocupa de la creación de programas y mecanismos que pueden mostrar comportamientos que podrían considerarse inteligentes.

## Hitos en la Historia de la IA

- **1956:** John McCarthy acuña el término "inteligencia artificial" en la conferencia de Dartmouth College.
- **1958:** Herbert Simon y Allen Newell desarrollan el programa de lógica simbólica "Logic Theorist".
- **1965:** Joseph Weizenbaum crea el programa de procesamiento de lenguaje natural "ELIZA".
- **1979:** Douglas Lenat desarrolla el programa de razonamiento "AM".
- **1997:** Deep Blue, de IBM, derrota al campeón mundial de ajedrez Garry Kasparov.
- **2011:** IBM Watson gana el concurso de televisión "Jeopardy!".
- **2016:** AlphaGo, de DeepMind, derrota al campeón mundial de Go, Lee Sedol.
- **2020:** AlphaFold, de DeepMind, predice la estructura de proteínas con una precisión sin precedentes.
- **2021:** GPT-3, de OpenAI, es lanzado y muestra un rendimiento sobresaliente en tareas de procesamiento del lenguaje natural.
- **2023:** GNoMe, de DeepMind, permite predecir materiales con propiedades específicas a partir de su estructura atómica y estructuras cristalinas.

## Primeras Técnicas Utilizadas en IA

### Lógica y Reglas Heurísticas

Se utilizaron reglas de lógica y heurísticas para representar el conocimiento y las estrategias de resolución de problemas. Estas reglas se aplicaron en sistemas expertos tempranos para modelar el razonamiento humano en dominios específicos.

### Árboles de Búsqueda y Algoritmos de Búsqueda

Se desarrollaron algoritmos de búsqueda como el algoritmo de búsqueda en anchura y el algoritmo de búsqueda en profundidad para encontrar soluciones a problemas mediante la exploración de un espacio de estados.

### Redes Neuronales Artificiales (ANN)

Aunque las ideas detrás de las redes neuronales se originaron en la década de 1940, se utilizaron más ampliamente en las décadas de 1950 y 1960 para modelar el comportamiento de las neuronas y para abordar problemas de aprendizaje automático y reconocimiento de patrones.

### Sistemas Expertos

Los sistemas expertos, que representaban el conocimiento en forma de reglas if-then y utilizaban motores de inferencia para razonar sobre ese conocimiento, fueron una de las primeras aplicaciones prácticas de la IA en campos como la medicina, la ingeniería y el diagnóstico.

### Programación Simbólica

Se desarrollaron lenguajes de programación simbólica como Lisp para manipular símbolos y representar el conocimiento de una manera más abstracta, lo que facilitaba la implementación de sistemas inteligentes.

### Teoría de Juegos y Planificación

Se aplicaron principios de teoría de juegos y planificación para desarrollar agentes inteligentes capaces de tomar decisiones en entornos complejos y competitivos.

### Aprendizaje Automático Simbólico

Surgieron enfoques de aprendizaje automático basados en el razonamiento simbólico, como el aprendizaje inductivo, que se centraba en la extracción de reglas y patrones a partir de ejemplos de datos.

## Diferencia entre Machine Learning y Deep Learning

### Machine Learning (ML)

El Machine Learning se refiere a un conjunto de técnicas que permiten a los ordenadores aprender patrones o realizar tareas específicas sin ser explícitamente programadas para ello.

- Se basa en algoritmos que pueden aprender de datos y hacer predicciones o tomar decisiones basadas en esos datos. Estos algoritmos pueden ser supervisados, no supervisados o de aprendizaje por refuerzo.
- Ejemplos de técnicas de Machine Learning incluyen regresión lineal, árboles de decisión, máquinas de vectores de soporte (Support Vector Machines), k-means clustering, entre otros.
- En términos de arquitectura, los modelos de Machine Learning pueden tener una o unas pocas capas, pero no son tan profundas o complejas como las redes neuronales profundas utilizadas en Deep Learning.

### Deep Learning (DL)

El Deep Learning es una subárea del Machine Learning que se centra en el uso de algoritmos basados en redes neuronales artificiales con múltiples capas (a menudo muchas capas) para modelar y procesar datos.

- Estas redes neuronales profundas están compuestas por múltiples capas de nodos interconectados que procesan la información de manera jerárquica, extrayendo características complejas de los datos de entrada.
- El Deep Learning se ha vuelto muy popular y efectivo en áreas como el reconocimiento de imágenes, el procesamiento del lenguaje natural, la visión por computadora y otros problemas de percepción.
- A diferencia de muchas técnicas de Machine Learning tradicionales, el Deep Learning requiere grandes cantidades de datos de entrenamiento y potencia computacional para ajustar correctamente los muchos parámetros de las redes neuronales profundas.

## Tipos de Redes Neuronales

### Redes Neuronales Artificiales (ANN)

Son la forma más básica de redes neuronales, compuestas por capas de neuronas conectadas. Cada neurona está conectada a las neuronas de la capa siguiente.

- Las capas de una red neuronal artificial pueden ser de tres tipos: capa de entrada, capas ocultas y capa de salida. Las capas ocultas son las capas intermedias entre la capa de entrada y la capa de salida.
- La capa de entrada recibe los datos de entrada, la capa de salida produce los resultados y las capas ocultas realizan el procesamiento intermedio.
- La capa de entrada define la dimensión de los datos de entrada, la capa de salida define la dimensión de los datos de salida y las capas ocultas definen la complejidad y la capacidad de aprendizaje del modelo.

### Redes Neuronales Convolucionales (CNN)

Especialmente diseñadas para procesar datos con estructura de cuadrícula, como imágenes. Utilizan operaciones de convolución para extraer características importantes de los datos de entrada.

- Las CNN son capaces de capturar patrones espaciales y de escala en las imágenes, lo que las hace muy efectivas en tareas de visión artificial, como la clasificación de imágenes, la detección de objetos y la segmentación semántica.

### Redes Neuronales Recurrentes (RNN)

Son adecuadas para datos de secuencia, como texto o series temporales. Tienen conexiones de retroalimentación que les permiten mantener y usar información a lo largo del tiempo.

- Originalmente, las RNN fueron útiles en tareas como el procesamiento del lenguaje natural, la traducción automática, la generación de texto y la predicción de series temporales. En la actualidad, han sido reemplazadas en muchos casos por las redes neuronales LSTM y transformers.
- El problema principal de las RNN es el desvanecimiento del gradiente, que dificulta el entrenamiento de redes grandes. El desvanecimiento del gradiente ocurre cuando los gradientes se vuelven muy pequeños a medida que se propagan hacia atrás en el tiempo, lo que dificulta la actualización de los pesos de las capas anteriores.

### Redes Neuronales Long Short-Term Memory (LSTM)

Una variante de las RNN diseñada para manejar problemas de desvanecimiento del gradiente. Las LSTM tienen unidades de memoria especiales que pueden aprender y recordar a largo plazo.

- Las LSTM son ampliamente utilizadas en tareas de procesamiento del lenguaje natural, como la traducción automática, la generación de texto y la generación de subtítulos de imágenes.

### Redes Neuronales Generativas Adversarias (GAN)

Consisten en dos redes neuronales, un generador y un discriminador, que compiten entre sí. El generador crea datos nuevos que intentan pasar como datos reales, mientras que el discriminador intenta distinguir entre los datos reales y los generados.

- Las GAN supusieron un gran avance en la generación de datos realistas y se utilizan en tareas de generación de imágenes, video y audio, así como en la mejora de la calidad de las imágenes y la generación de datos sintéticos.

### Redes Neuronales Siamesas

Utilizadas en tareas de comparación o identificación de similitudes. Consisten en dos ramas de redes neuronales que comparten los mismos parámetros y procesan dos entradas para producir vectores de características que luego se comparan.

### Redes Neuronales Autoencoder

Utilizadas para el aprendizaje no supervisado, comprimen los datos de entrada en un espacio de representación más pequeño y luego los reconstruyen. Pueden ser utilizadas para la reducción de dimensionalidad, la generación de datos y la detección de anomalías.

### Redes Neuronales Residuales (ResNet)

Introducen conexiones de "salto" que permiten que las señales de entrada y salida se agreguen directamente entre capas. Esto facilita el entrenamiento de redes más profundas al evitar problemas de desvanecimiento del gradiente.

### Redes Neuronales Transformer

Introducen un mecanismo de atención que permite a las redes neuronales procesar secuencias de datos de manera paralela y capturar relaciones a largo plazo entre elementos de la secuencia.

- Los transformers han demostrado ser muy efectivos en tareas de procesamiento del lenguaje natural, como la traducción automática, la generación de texto y la respuesta a preguntas.
- La gran ventaja de los transformers es su capacidad para capturar relaciones a largo plazo en las secuencias de datos, y permiten ser entrenados de manera más escalable y eficiente que modelos anteriores.

### Redes Neuronales de difusores

Son un tipo de red neuronal generativa que modela la distribución de probabilidad de los datos de entrada. Utilizan una serie de transformaciones invertibles para mapear los datos de entrada a un espacio latente y viceversa.

### Redes Neuronales de aprendizaje por refuerzo (RL)

Se utilizan para entrenar agentes inteligentes que toman decisiones secuenciales en entornos dinámicos. Los agentes aprenden a maximizar una recompensa acumulada a lo largo del tiempo.

- El campo del aprendizaje por refuerzo ha experimentado un gran avance en los últimos años, con el desarrollo de algoritmos como DQN, A2C, PPO y DDPG, que han demostrado un rendimiento sobresaliente en tareas de control y juegos.

## Tipos de modelos de IA

### Modelos de Aprendizaje Supervisado

Los modelos de aprendizaje supervisado se entrenan con ejemplos de entrada y salida emparejados. El objetivo es aprender una función que mapee las entradas a las salidas.

- Ejemplos de modelos de aprendizaje supervisado incluyen regresión lineal, regresión logística, máquinas de vectores de soporte (SVM), árboles de decisión, bosques aleatorios, redes neuronales, entre otros.
- Estos modelos se utilizan en tareas como la clasificación, la regresión, la detección de anomalías y la generación de texto.

### Modelos de Aprendizaje No Supervisado

Los modelos de aprendizaje no supervisado se entrenan con datos de entrada sin etiquetar. El objetivo es encontrar patrones, estructuras o relaciones interesantes en los datos.

- Ejemplos de modelos de aprendizaje no supervisado incluyen clustering, reducción de dimensionalidad, reglas de asociación y aprendizaje de densidad.
- Estos modelos se utilizan en tareas como la segmentación de clientes, la detección de fraudes, la recomendación de productos y la visualización de datos.

### Modelos de Aprendizaje por Refuerzo

Los modelos de aprendizaje por refuerzo se entrenan con un sistema de recompensa y castigo. El objetivo es aprender una política que maximice la recompensa acumulada a lo largo del tiempo.

- Ejemplos de modelos de aprendizaje por refuerzo incluyen Q-learning, SARSA, DQN, A2C, PPO y DDPG.
- Estos modelos se utilizan en tareas como el control de robots, los juegos, la optimización de carteras y la toma de decisiones secuenciales.

### Modelos de Aprendizaje Semi-Supervisado

Los modelos de aprendizaje semi-supervisado se entrenan con una combinación de datos etiquetados y no etiquetados. El objetivo es aprovechar la información no etiquetada para mejorar el rendimiento del modelo.

- Ejemplos de modelos de aprendizaje semi-supervisado incluyen la propagación de etiquetas, la autoetiquetación y la regularización de consistencia.
- Estos modelos se utilizan en tareas donde es costoso o difícil obtener grandes cantidades de datos etiquetados.

### Modelos de Aprendizaje por Transferencia

Los modelos de aprendizaje por transferencia se entrenan en un dominio fuente y se aplican en un dominio objetivo relacionado. El objetivo es transferir el conocimiento aprendido en el dominio fuente al dominio objetivo.

- Ejemplos de modelos de aprendizaje por transferencia incluyen fine-tuning, pre-entrenamiento y adaptación de dominio.
- Los modelos de aprendizaje por transferencia se utilizan en tareas donde hay poca cantidad de datos en el dominio objetivo o donde el entrenamiento desde cero es costoso.

## Modelos de IA pre-entrenados

Los modelos de IA pre-entrenados son modelos que han sido entrenados en grandes conjuntos de datos y que se pueden utilizar directamente o ajustar para tareas específicas.

### Conceptos Relacionados con Modelos de IA pre-entrenados

- **Checkpoints (Puntos de control):** Instantáneas guardadas del estado del modelo durante el proceso de entrenamiento en IA. Se utilizan para reanudar el entrenamiento o para realizar inferencias. En un checkpoint se guardan los pesos, los hiperparámetros y otros datos del modelo.
- **Transfer Learning (Aprendizaje por transferencia):** Técnica en la que un modelo entrenado para una tarea específica se reutiliza como punto de partida para entrenar otro modelo para una tarea relacionada o diferente.
- **Hypernetworks (Hiperredes):** Clase de modelos de redes neuronales utilizados para generar pesos o parámetros de otras redes neuronales. Se utilizan para aprender representaciones de datos o para generar arquitecturas de redes neuronales.
- **Data Augmentation (Aumento de datos):** Técnica para aumentar la cantidad y diversidad de datos de entrenamiento mediante transformaciones aleatorias o controladas.
- **Adversarial Training (Entrenamiento adversarial):** Técnica de entrenamiento para modelos generativos que implica entrenar simultáneamente un generador y un discriminador.
- **Self-Attention (Autoatención):** Mecanismo utilizado en arquitecturas de redes neuronales, especialmente en modelos de lenguaje como Transformers.
- **Latent Space (Espacio latente):** Espacio de representación de características latentes aprendidas por un modelo generativo.
- **Fine-Tuning (Ajuste fino):** Técnica de ajuste de un modelo pre-entrenado en un conjunto de datos específico para mejorar su rendimiento en una tarea específica.
- **Inference (Inferencia):** Proceso de utilizar un modelo entrenado para hacer predicciones sobre nuevos datos de entrada.

## Principales Librerías de Inteligencia Artificial

- **TensorFlow:** Desarrollada por Google, una de las librerías más populares para construir y entrenar modelos de IA y DL. [TensorFlow](https://www.tensorflow.org)
- **PyTorch:** Desarrollada por Facebook, muy popular en la investigación y el desarrollo de prototipos. [PyTorch](https://pytorch.org/)
- **Scikit-learn:** Librería de aprendizaje automático en Python para algoritmos supervisados y no supervisados. [Scikit-learn](https://scikit-learn.org/)
- **Keras:** Librería de alto nivel para la construcción de redes neuronales en Python que puede ejecutarse sobre TensorFlow, Theano o CNTK. [Keras](https://keras.io/)
- **MXNet:** Librería de código abierto para el desarrollo de modelos de IA y DL. [MXNet](https://mxnet.apache.org/)
- **Caffe:** Librería especialmente diseñada para visión por computadora y CNN. [Caffe](https://caffe.berkeleyvision.org/)
- **OpenCV:** Librería de visión por computadora de código abierto. [OpenCV](https://opencv.org/)
- **NLTK (Natural Language Toolkit):** Librería de Python para el procesamiento del lenguaje natural. [NLTK](https://www.nltk.org/)

## Enlaces de Interés

### Conceptos Básicos de IA

- [Inteligencia Artificial en Wikipedia](https://es.wikipedia.org/wiki/Inteligencia_artificial)
- [Historia de la Inteligencia Artificial](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence)
- [Tensorflow Playground](https://playground.tensorflow.org/)

### Empresas relevantes en IA

- [OpenAI](https://openai.com/)
- [DeepMind](https://deepmind.com/)
- [IBM Watson](https://www.ibm.com/watson)

### Plataformas y Comunidades

- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/)
- [Ollama](https://ollama.com/)
- [Civit AI](https://civitai.com/)

### Miscelánea

- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [KDnuggets](https://www.kdnuggets.com/)
- [arXiv](https://www.arxiv.org/)
- [Papers with Code](https://www.paperswithcode.com/)

## Anexo 1: Modelos de IA Pre-entrenados

### Modelos generativos de lenguaje

- **GPT-X (Generative Pre-trained Transformer X):** Modelo de lenguaje generativo de OpenAI.
- **[Llama](https://llama.meta.com/):** Modelo de lenguaje generativo de Meta, con muchas variantes y modelos especializados.
- **[gemma](https://ollama.com/library/gemma):** Modelo de lenguaje generativo de Google (Deepmind).
- **[Mistral](https://docs.mistral.ai/):** Modelo de lenguaje generativo de Mistral, con variantes como Mixtral y Dolphin-Mixtral.
- **[Qwen](https://github.com/QwenLM/Qwen):** Modelo de lenguaje generativo de Alibaba.
- **[Llava](https://llava-vl.github.io/):** Modelo multimodal que combina texto y visión.

### Modelos generativos de imágenes

- **[DALL-E 2](https://openai.com/dall-e-2/):** Modelo generativo de imágenes de OpenAI.
- **[MidJourney](https://www.midjourney.com/home):** Modelo generativo de imágenes de MidJourney.
- **[Stable Diffusion](https://stability.ai/stable-image):** Modelo generativo de imágenes de OpenAI.

## Anexo 2: Hitos y Logros de DeepMind

DeepMind es una empresa de inteligencia artificial con sede en Londres, fundada en 2010 y adquirida por Google en 2014. Ha logrado varios hitos y avances significativos en el campo de la inteligencia artificial.

### AlphaGo

- En 2016, AlphaGo, desarrollado por DeepMind, derrotó al campeón mundial de Go, Lee Sedol, en una serie de juegos históricos. AlphaGo demostró la capacidad de las redes neuronales para dominar un juego complejo de estrategia, superando el nivel humano.
- [AlphaGo - The Movie](https://youtu.be/WXuK6gekU1Y)

### AlphaZero

- En 2017, DeepMind presentó AlphaZero, un sistema de IA capaz de aprender a jugar Go, ajedrez y shogi sin datos de entrenamiento humanos.
- [How Magnus Carlsen Learned From AlphaZero](https://youtu.be/I0zqbO622rg)

### AlphaStar

- En 2019, DeepMind presentó AlphaStar, un sistema de IA capaz de jugar StarCraft II a nivel de los mejores jugadores humanos.
- [AlphaStar - The inside story](https://youtu.be/UuhECwm31dM)

### AlphaFold

- En 2020, DeepMind presentó AlphaFold, un sistema de IA para la predicción de la estructura de proteínas.
- AlphaFold demostró una capacidad sin precedentes para predecir la estructura tridimensional de las proteínas, un avance significativo en la biología computacional.
- [AlphaFold en YouTube](https://youtube.com/playlist?list=PLqYmG7hTraZAhkAh72kzzLC4r2O4VoVgz)

### GNoMe

- En 2023, DeepMind presentó GNoMe, un modelo que permite predecir materiales con propiedades específicas a partir de su estructura atómica y estructuras cristalinas.
- [Millions of new materials discovered with deep learning](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/)
