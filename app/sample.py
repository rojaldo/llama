import whisper
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering

# Configuración inicial
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda")
)

def transcribir_con_diarizacion(audio_path, num_hablantes=2):
    # Transcripción con Whisper
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    segments = result["segments"]

    # Extracción de embeddings
    embeddings = []
    for segment in segments:
        waveform = extraer_audio_segmento(audio_path, segment["start"], segment["end"])
        embedding = embedding_model(waveform)
        embeddings.append(embedding.numpy())
    
    # Clustering de hablantes
    clustering = AgglomerativeClustering(n_clusters=num_hablantes).fit(embeddings)
    labels = clustering.labels_
    
    # Asignación de etiquetas
    for i, segment in enumerate(segments):
        segment["speaker"] = f"Hablante {labels[i]+1}"
    
    return segments

def extraer_audio_segmento(audio_path, start, end):
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)         # [channels, samples]
    waveform = waveform.unsqueeze(0)                    # [1, channels, samples]
    s = int(start * sr)
    e = int(end   * sr)
    return waveform[:, :, s:e]

if __name__ == "__main__":
    audio_path = "./docs/audio.mp4"  
    num_hablantes = 2
    segmentos = transcribir_con_diarizacion(audio_path, num_hablantes)
    
    for segmento in segmentos:
        print(f"{segmento['speaker']}: {segmento['text']} (desde {segmento['start']}s hasta {segmento['end']}s)")
