from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import soundfile as sf
import numpy as np

model_name = "./uzbek-stt-3"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Use CPU
device = torch.device("cpu")
model.to(device)

def load_and_preprocess_audio(file_path):
    # Use soundfile to load the audio file
    speech_array, sampling_rate = sf.read(file_path)
    if sampling_rate != 16000:
        # Convert to torch tensor for resampling
        speech_tensor = torch.FloatTensor(speech_array)
        if len(speech_tensor.shape) > 1:
            speech_tensor = speech_tensor.mean(dim=1)  # Convert stereo to mono
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = resampler(speech_tensor)
        speech_array = speech_tensor.numpy()
    return speech_array

def replace_unk(transcription):
    return transcription.replace("[UNK]", "ʼ")

def chunk_audio(speech_array, chunk_size, overlap=0):
    """
    Divide audio into chunks with optional overlap.
    """
    step_size = chunk_size - overlap
    chunks = []
    for start in range(0, len(speech_array) - chunk_size + 1, step_size):
        chunks.append(speech_array[start : start + chunk_size])
    if len(speech_array) % step_size != 0:
        chunks.append(speech_array[-chunk_size:])  # Add the last chunk
    return chunks

audio_file = "techno.wav"
speech_array = load_and_preprocess_audio(audio_file)

# Define chunk size (e.g., 5 seconds of audio) and overlap (e.g., 1 second)
chunk_size = int(5 * 16000)  # 5 seconds at 16kHz
overlap = int(1 * 16000)     # 1 second overlap

chunks = chunk_audio(speech_array, chunk_size, overlap)

# Transcribe each chunk and concatenate results
transcription_text = ""
for chunk in chunks:
    input_values = processor(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    transcription_text += replace_unk(transcription[0]) + " "
    print("Transcription:", transcription_text.strip())
