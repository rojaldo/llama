
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np

# Configuración inicial
ARCHIVO_AUDIO = "data/Neil.mp3"
MODELO_WHISPER = "large-v3"
TOKEN_HUGGINGFACE = ""  # Reemplazar con tu token real

def procesar_audio(archivo):
    # Convertir a formato WAV y frecuencia estándar
    audio = AudioSegment.from_file(archivo)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export("temp.wav", format="wav")
    return "temp.wav"

def transcribir_whisper(archivo_wav):
    model = whisper.load_model(MODELO_WHISPER)
    result = model.transcribe(archivo_wav)
    return result["segments"]

def diarizar_hablantes(archivo_wav):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=TOKEN_HUGGINGFACE)
    
    diarizacion = pipeline(archivo_wav)
    return diarizacion

def combinar_resultados(segmentos, diarizacion):
    resultado = []
    for segmento in segmentos:
        inicio = segmento["start"]
        fin = segmento["end"]
        texto = segmento["text"]
        
        # Encontrar hablante predominante en el segmento
        hablante = max(
            [(intervalo, label) for intervalo, _, label in diarizacion.itertracks(yield_label=True)
             if intervalo.start <= inicio and intervalo.end >= fin],
            key=lambda x: min(fin, x[0].end) - max(inicio, x[0].start),
            default=(None, "Desconocido")
        )[1]
        
        resultado.append({
            "inicio": inicio,
            "fin": fin,
            "hablante": hablante,
            "texto": texto
        })
    
    return resultado

# Flujo principal
audio_procesado = procesar_audio(ARCHIVO_AUDIO)
transcripcion = transcribir_whisper(audio_procesado)
diarizacion = diarizar_hablantes(audio_procesado)
resultado_final = combinar_resultados(transcripcion, diarizacion)

# Mostrar resultados
for entrada in resultado_final:
    print(f"[{entrada['inicio']:.1f}s - {entrada['fin']:.1f}s] {entrada['hablante']}:")
    print(f"    {entrada['texto']}\n")
