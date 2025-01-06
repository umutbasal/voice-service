from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import numpy as np
import re


model = VitsModel.from_pretrained("./mms-tts-uzb-script_cyrillic")
tokenizer = AutoTokenizer.from_pretrained("./mms-tts-uzb-script_cyrillic")

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

text_uzbek_cyrilic = "Салом,  хуш келибсиз!  Сиз Аҳмет Ин витро уруғлантириш ва репродуктив саломатлик марказига улангансиз,  мен Алиман.  Сизга қандай ёрдам беришим мумкин? Клиникамизда проф.  Др.  Аҳмет Каража ва жамоамиз билан энг замонавий даволаш усулларини таклиф этамиз.  Агар сиз учрашувга ёзилиш,  жараёнлар ҳақида билиш ёки маслаҳат хизматига мурожаат қилишни истасангиз,  биз сизга энг яхши тарзда ёрдам беришдан хурсанд бўламиз.  Мен сизни тинглаш учун келдим,  саволларингизга жавоб бера оламан. . . . "

# Get text chunks
chunks = split_into_chunks(text_uzbek_cyrilic)
all_waveforms = []

with torch.no_grad():
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}: {chunk[:50]}...")
        
        # Tokenize the chunk
        inputs = tokenizer(chunk, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()
        
        try:
            # Generate audio for this chunk
            output = model(**inputs).waveform
            
            # Save this chunk
            chunk_path = f"chunk_{i+1}.wav"
            scipy.io.wavfile.write(chunk_path, 
                                 rate=model.config.sampling_rate,
                                 data=output.numpy().squeeze())
            
            all_waveforms.append(output.squeeze())
            print(f"Generated chunk {i+1}")
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue

    if all_waveforms:
        # Combine all chunks
        combined_waveform = torch.cat(all_waveforms, dim=0)
        
        # Save the complete audio
        scipy.io.wavfile.write("techno_streamed.wav", 
                             rate=model.config.sampling_rate,
                             data=combined_waveform.numpy())
