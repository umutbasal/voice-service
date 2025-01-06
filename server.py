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
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize TTS models
logger.info("Initializing TTS models...")
tts_model = VitsModel.from_pretrained("./mms-tts-uzb-script_cyrillic")
tts_tokenizer = AutoTokenizer.from_pretrained("./mms-tts-uzb-script_cyrillic")
logger.info("TTS models initialized successfully")

# Initialize STT models
logger.info("Initializing STT models...")
stt_model = Wav2Vec2ForCTC.from_pretrained("./uzbek-stt-3")
stt_processor = Wav2Vec2Processor.from_pretrained("./uzbek-stt-3")
logger.info("STT models initialized successfully")

def clean_text(text):
    logger.debug(f"Cleaning text: {text[:50]}...")
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'\.+', '.', cleaned)
    logger.debug(f"Text cleaned: {cleaned[:50]}...")
    return cleaned.strip()

def split_into_chunks(text, max_length=100):
    logger.debug(f"Splitting text into chunks, max_length={max_length}")
    text = clean_text(text)
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
    
    logger.debug(f"Split into {len(chunks)} chunks")
    return chunks

def replace_unk(transcription):
    logger.debug("Replacing [UNK] tokens in transcription")
    return transcription.replace("[UNK]", "ʼ")

def load_and_preprocess_audio(audio_bytes):
    logger.debug("Loading and preprocessing audio")
    speech_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
    
    if len(speech_array.shape) > 1:
        logger.debug("Converting stereo to mono")
        speech_array = speech_array.mean(axis=1)
    
    if sampling_rate != 16000:
        logger.debug(f"Resampling audio from {sampling_rate}Hz to 16000Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = torch.FloatTensor(speech_array)
        speech_array = resampler(speech_tensor).numpy()
    
    logger.debug("Audio preprocessing completed")
    return speech_array

class TextToSpeechServicer(voice_pb2_grpc.TextToSpeechServiceServicer):
    def UnaryConvertTextToSpeech(self, request, context):
        logger.info("Received UnaryConvertTextToSpeech request")
        logger.debug(f"Input text: {request.text[:50]}...")
        
        inputs = tts_tokenizer(request.text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()
        
        try:
            with torch.no_grad():
                output = tts_model(**inputs).waveform
                
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
            logger.info("Successfully converted text to speech")
            return voice_pb2.TextToSpeechResponse(audio_data=audio_bytes.getvalue())
        except Exception as e:
            logger.error(f"Error in UnaryConvertTextToSpeech: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def ServerStreamTextToSpeech(self, request, context):
        logger.info("Starting ServerStreamTextToSpeech")
        logger.debug(f"Input text: {request.text[:50]}...")
        chunks = split_into_chunks(request.text)
        logger.info(f"Processing {len(chunks)} text chunks")
        
        try:
            with torch.no_grad():
                for i, chunk in enumerate(chunks, 1):
                    logger.debug(f"Processing chunk {i}/{len(chunks)}")
                    inputs = tts_tokenizer(chunk, return_tensors="pt")
                    inputs['input_ids'] = inputs['input_ids'].long()
                    
                    output = tts_model(**inputs).waveform
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
                    
                    yield voice_pb2.AudioChunk(audio_data=audio_bytes.getvalue())
            logger.info("Successfully completed ServerStreamTextToSpeech")
        except Exception as e:
            logger.error(f"Error in ServerStreamTextToSpeech: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def ClientStreamTextToSpeech(self, request_iterator, context):
        logger.info("Starting ClientStreamTextToSpeech")
        all_waveforms = []
        chunk_count = 0
        
        try:
            with torch.no_grad():
                for text_chunk in request_iterator:
                    chunk_count += 1
                    logger.debug(f"Processing chunk {chunk_count}")
                    inputs = tts_tokenizer(text_chunk.text, return_tensors="pt")
                    inputs['input_ids'] = inputs['input_ids'].long()
                    
                    output = tts_model(**inputs).waveform
                    all_waveforms.append(output.squeeze())
            
            logger.debug("Combining all waveforms")
            combined_waveform = torch.cat(all_waveforms, dim=0)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, combined_waveform.numpy(), tts_model.config.sampling_rate, format='WAV')
            
            logger.info(f"Successfully processed {chunk_count} chunks")
            return voice_pb2.TextToSpeechResponse(audio_data=audio_bytes.getvalue())
        except Exception as e:
            logger.error(f"Error in ClientStreamTextToSpeech: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def BidirectionalStreamTextToSpeech(self, request_iterator, context):
        logger.info("Starting BidirectionalStreamTextToSpeech")
        chunk_count = 0
        
        try:
            with torch.no_grad():
                for text_chunk in request_iterator:
                    chunk_count += 1
                    logger.debug(f"Processing chunk {chunk_count}")
                    inputs = tts_tokenizer(text_chunk.text, return_tensors="pt")
                    inputs['input_ids'] = inputs['input_ids'].long()
                    
                    output = tts_model(**inputs).waveform
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, output.numpy().squeeze(), tts_model.config.sampling_rate, format='WAV')
                    
                    yield voice_pb2.AudioChunk(audio_data=audio_bytes.getvalue())
            logger.info(f"Successfully processed {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Error in BidirectionalStreamTextToSpeech: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

class SpeechToTextServicer(voice_pb2_grpc.SpeechToTextServiceServicer):
    def UnaryConvertSpeechToText(self, request, context):
        logger.info("Starting UnaryConvertSpeechToText")
        
        try:
            speech_array = load_and_preprocess_audio(request.audio_data)
            input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = stt_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_processor.batch_decode(predicted_ids)
                
            text = replace_unk(transcription[0])
            logger.info("Successfully converted speech to text")
            logger.debug(f"Transcribed text: {text[:50]}...")
            return voice_pb2.SpeechToTextResponse(transcribed_text=text)
        except Exception as e:
            logger.error(f"Error in UnaryConvertSpeechToText: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def ServerStreamSpeechToText(self, request, context):
        logger.info("Starting ServerStreamSpeechToText")
        
        try:
            speech_array = load_and_preprocess_audio(request.audio_data)
            chunk_size = int(5 * 16000)
            overlap = int(1 * 16000)
            total_chunks = (len(speech_array) - overlap) // (chunk_size - overlap)
            logger.info(f"Processing audio in {total_chunks} chunks")
            
            for i in range(0, len(speech_array), chunk_size - overlap):
                logger.debug(f"Processing chunk starting at {i}")
                chunk = speech_array[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                input_values = stt_processor(chunk, sampling_rate=16000, return_tensors="pt").input_values
                
                with torch.no_grad():
                    logits = stt_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = stt_processor.batch_decode(predicted_ids)
                    
                text = replace_unk(transcription[0])
                logger.debug(f"Chunk transcription: {text[:50]}...")
                yield voice_pb2.TranscriptionChunk(partial_transcription=text)
            
            logger.info("Successfully completed streaming transcription")
        except Exception as e:
            logger.error(f"Error in ServerStreamSpeechToText: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def ClientStreamSpeechToText(self, request_iterator, context):
        logger.info("Starting ClientStreamSpeechToText")
        all_audio = []
        chunk_count = 0
        
        try:
            for chunk in request_iterator:
                chunk_count += 1
                logger.debug(f"Processing audio chunk {chunk_count}")
                chunk_array = load_and_preprocess_audio(chunk.audio_data)
                all_audio.append(chunk_array)
            
            logger.debug("Combining all audio chunks")
            speech_array = np.concatenate(all_audio)
            input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = stt_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_processor.batch_decode(predicted_ids)
                
            text = replace_unk(transcription[0])
            logger.info(f"Successfully processed {chunk_count} chunks")
            logger.debug(f"Final transcription: {text[:50]}...")
            return voice_pb2.SpeechToTextResponse(transcribed_text=text)
        except Exception as e:
            logger.error(f"Error in ClientStreamSpeechToText: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
    
    def BidirectionalStreamSpeechToText(self, request_iterator, context):
        logger.info("Starting BidirectionalStreamSpeechToText")
        chunk_buffer = []
        chunk_size = int(5 * 16000)
        chunk_count = 0
        
        try:
            for chunk in request_iterator:
                chunk_count += 1
                logger.debug(f"Received chunk {chunk_count}")
                chunk_array = load_and_preprocess_audio(chunk.audio_data)
                chunk_buffer.append(chunk_array)
                
                combined_length = sum(len(c) for c in chunk_buffer)
                if combined_length >= chunk_size:
                    logger.debug("Processing buffered chunks")
                    speech_array = np.concatenate(chunk_buffer)
                    input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
                    
                    with torch.no_grad():
                        logits = stt_model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = stt_processor.batch_decode(predicted_ids)
                    
                    text = replace_unk(transcription[0])
                    logger.debug(f"Chunk transcription: {text[:50]}...")
                    yield voice_pb2.TranscriptionChunk(partial_transcription=text)
                    chunk_buffer = []
            
            logger.info(f"Successfully processed {chunk_count} chunks")

            if chunk_buffer:
                logger.debug("Processing remaining buffered chunks")
                speech_array = np.concatenate(chunk_buffer)
                input_values = stt_processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
                
                with torch.no_grad():
                    logits = stt_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = stt_processor.batch_decode(predicted_ids)

                text = replace_unk(transcription[0])
                logger.debug(f"Remaining transcription: {text[:50]}...")
                yield voice_pb2.TranscriptionChunk(partial_transcription=text)

            logger.info(f"Successfully processed {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Error in BidirectionalStreamSpeechToText: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

def serve():
    logger.info("Starting gRPC server")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_pb2_grpc.add_TextToSpeechServiceServicer_to_server(TextToSpeechServicer(), server)
    voice_pb2_grpc.add_SpeechToTextServiceServicer_to_server(SpeechToTextServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 