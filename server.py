import grpc
from concurrent import futures
import voice_pb2
import voice_pb2_grpc
import torch
import io
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
import re

# Initialize TTS models
tts_model = VitsModel.from_pretrained("./mms-tts-uzb-script_cyrillic")
tts_tokenizer = AutoTokenizer.from_pretrained("./mms-tts-uzb-script_cyrillic")

# Initialize STT models
stt_model = Wav2Vec2ForCTC.from_pretrained("./uzbek-stt-3")
stt_processor = Wav2Vec2Processor.from_pretrained("./uzbek-stt-3")

def clean_text(text):
    # Remove multiple spaces and dots
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    return text.strip()

def split_into_chunks(text, max_length=100):
    # Clean the text first
    text = clean_text(text)
    
    # Split on sentence endings
    sentences = re.split('([.!?])', text)
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        sentence = sentence.strip()
        
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def replace_unk(transcription):
    return transcription.replace("[UNK]", "ʼ")

def load_and_preprocess_audio(audio_bytes):
    # Load audio from bytes
    speech_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Convert stereo to mono if needed
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = torch.FloatTensor(speech_array)
        speech_array = resampler(speech_tensor).numpy()
    
    return speech_array

class TextToSpeechServicer(voice_pb2_grpc.TextToSpeechServiceServicer):
    def ConvertTextToSpeech(self, request, context):
        """Simple: Text in, Audio out"""
        inputs = tts_tokenizer(request.text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()
        
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            
        # Convert to bytes
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
        return voice_pb2.TextToSpeechResponse(audio_data=audio_bytes.getvalue())
    
    def StreamTextToSpeech(self, request, context):
        """Stream response: Long text in, stream of audio chunks out"""
        chunks = split_into_chunks(request.text)
        
        with torch.no_grad():
            for chunk in chunks:
                inputs = tts_tokenizer(chunk, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].long()
                
                output = tts_model(**inputs).waveform
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
                
                yield voice_pb2.AudioChunk(audio_data=audio_bytes.getvalue())
    
    def StreamTextToSpeechRequest(self, request_iterator, context):
        """Stream request: Stream of text chunks in, complete audio out"""
        all_waveforms = []
        
        with torch.no_grad():
            for text_chunk in request_iterator:
                inputs = tts_tokenizer(text_chunk.text, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].long()
                
                output = tts_model(**inputs).waveform
                all_waveforms.append(output.squeeze())
        
        # Combine all waveforms
        combined_waveform = torch.cat(all_waveforms, dim=0)
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, combined_waveform.numpy(), tts_model.config.sampling_rate, format='WAV')
        
        return voice_pb2.TextToSpeechResponse(audio_data=audio_bytes.getvalue())
    
    def StreamTextToSpeechBidirectional(self, request_iterator, context):
        """Bidirectional: Stream of text chunks in, stream of audio chunks out"""
        with torch.no_grad():
            for text_chunk in request_iterator:
                inputs = tts_tokenizer(text_chunk.text, return_tensors="pt")
                inputs['input_ids'] = inputs['input_ids'].long()
                
                output = tts_model(**inputs).waveform
                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
                
                yield voice_pb2.AudioChunk(audio_data=audio_bytes.getvalue())

class SpeechToTextServicer(voice_pb2_grpc.SpeechToTextServiceServicer):
    def ConvertSpeechToText(self, request, context):
        """Simple: Audio in, Text out"""
        speech_array = load_and_preprocess_audio(request.audio_data)
        input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
        
        with torch.no_grad():
            logits = stt_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = stt_processor.batch_decode(predicted_ids)
            
        text = replace_unk(transcription[0])
        return voice_pb2.SpeechToTextResponse(transcribed_text=text)
    
    def StreamSpeechToText(self, request, context):
        """Stream response: Complete audio in, stream of text chunks out"""
        speech_array = load_and_preprocess_audio(request.audio_data)
        
        # Process in chunks
        chunk_size = int(5 * 16000)  # 5 seconds
        overlap = int(1 * 16000)     # 1 second overlap
        
        for i in range(0, len(speech_array), chunk_size - overlap):
            chunk = speech_array[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk if needed
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            input_values = stt_processor(chunk, sampling_rate=16000, return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = stt_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_processor.batch_decode(predicted_ids)
                
            text = replace_unk(transcription[0])
            yield voice_pb2.TranscriptionChunk(partial_transcription=text)
    
    def StreamSpeechToTextRequest(self, request_iterator, context):
        """Stream request: Stream of audio chunks in, complete text out"""
        all_audio = []
        
        # Collect all audio chunks
        for chunk in request_iterator:
            chunk_array = load_and_preprocess_audio(chunk.audio_data)
            all_audio.append(chunk_array)
        
        # Combine all chunks
        speech_array = np.concatenate(all_audio)
        input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
        
        with torch.no_grad():
            logits = stt_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = stt_processor.batch_decode(predicted_ids)
            
        text = replace_unk(transcription[0])
        return voice_pb2.SpeechToTextResponse(transcribed_text=text)
    
    def StreamSpeechToTextBidirectional(self, request_iterator, context):
        """Bidirectional: Stream of audio chunks in, stream of text chunks out"""
        chunk_buffer = []
        chunk_size = int(5 * 16000)  # 5 seconds worth of audio
        
        for chunk in request_iterator:
            chunk_array = load_and_preprocess_audio(chunk.audio_data)
            chunk_buffer.append(chunk_array)
            
            # Process when we have enough audio data
            combined_length = sum(len(c) for c in chunk_buffer)
            if combined_length >= chunk_size:
                # Combine buffered chunks
                speech_array = np.concatenate(chunk_buffer)
                input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
                
                with torch.no_grad():
                    logits = stt_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = stt_processor.batch_decode(predicted_ids)
                
                text = replace_unk(transcription[0])
                yield voice_pb2.TranscriptionChunk(partial_transcription=text)
                
                # Reset buffer
                chunk_buffer = []

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_pb2_grpc.add_TextToSpeechServiceServicer_to_server(TextToSpeechServicer(), server)
    voice_pb2_grpc.add_SpeechToTextServiceServicer_to_server(SpeechToTextServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 