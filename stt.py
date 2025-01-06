from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import soundfile as sf
import numpy as np

model_name = "./uzbek-stt-3"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Use CPU instead of MPS
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

audio_file = "techno.wav"
speech_array = load_and_preprocess_audio(audio_file)

input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

transcription_text = replace_unk(transcription[0])

print("Transcription:", transcription_text)
