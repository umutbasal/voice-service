from transformers import VitsModel, AutoTokenizer
import torch
import scipy


model = VitsModel.from_pretrained("./mms-tts-uzb-script_cyrillic")
tokenizer = AutoTokenizer.from_pretrained("./mms-tts-uzb-script_cyrillic")

text_uzbek_cyrilic = "Салом,  хуш келибсиз!  Сиз Аҳмет Ин витро уруғлантириш ва репродуктив саломатлик марказига улангансиз,  мен Алиман.  Сизга қандай ёрдам беришим мумкин? Клиникамизда проф.  Др.  Аҳмет Каража ва жамоамиз билан энг замонавий даволаш усулларини таклиф этамиз.  Агар сиз учрашувга ёзилиш,  жараёнлар ҳақида билиш ёки маслаҳат хизматига мурожаат қилишни истасангиз,  биз сизга энг яхши тарзда ёрдам беришдан хурсанд бўламиз.  Мен сизни тинглаш учун келдим,  саволларингизга жавоб бера оламан. . . . "
inputs = tokenizer(text_uzbek_cyrilic, return_tensors="pt")

# Convert input_ids to Long tensor type
inputs['input_ids'] = inputs['input_ids'].long()
if 'attention_mask' in inputs:
    inputs['attention_mask'] = inputs['attention_mask'].long()

with torch.no_grad():
    #use waveform_chunks for streaming
    output = model(**inputs).waveform

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.numpy().squeeze())
