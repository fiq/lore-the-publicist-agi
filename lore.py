import shutup;
shutup.please()

import random
from bark.generation import load_codec_model, generate_text_semantic
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from bark.api import generate_audio
from encodec.utils import convert_audio
import torchaudio
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from transformers import BertTokenizer
import sounddevice as sd
import pprint
import whisper
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline
generator = pipeline('text-generation', model='gpt2')
preload_models()
#tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=tokenizer.eos_token_id)

pp = pprint.PrettyPrinter()
DEVICE = "mps"
LISTEN_FILE = "./hearing.wav"
IMPERFECTIONS = ['[laughs]', '[sings]', '[exclaims]', 'lala', 'Mmmm', 'hmm', '🎵']

model = whisper.load_model("base")
voice_name="en_speaker_4"


def imperfectionise(thought_stream):
    words = thought_stream.split(' ')
    for i in range(10):
        imperfection = IMPERFECTIONS[random.randint(0,len(IMPERFECTIONS)-1)]
        cut_point = random.randint(0, len(words)-1)
        words = words[0:cut_point] + [ imperfection ] + words[cut_point:]
    return " ".join(words)

def vocalise(audio_array):
    pp.pprint(audio_array)
    write_wav("output.wav", SAMPLE_RATE, audio_array)
    sd.play(audio_array, SAMPLE_RATE)
    # allow async sd.play to complete
    sd.wait()  

def comprehend_in_song(text):
    prompt = "Here's a catchy song to publicise " + text +":\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = gpt_neo_model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=120
    )
    generated_response = tokenizer.batch_decode(gen_tokens)[0]
    return "\n".join( generated_response.split("\n")[2:] )

def hearing():
    result = model.transcribe(LISTEN_FILE)
    return result["text"].strip()

def listen():
    duration = 4  # seconds
    hearing_in = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write_wav(LISTEN_FILE, SAMPLE_RATE, hearing_in)
    return hearing()

def sentience():
    while True:
        input("Press enter to create a new jingle")
        print("\n\nWhat cool thing should we publicise? Speak now.... Yes. Speak out loud. I'm listening")
        text = listen()
        print("I heard: " + text)
        response = comprehend_in_song(text)
        imperfect_response = imperfectionise( response )
        print("Here's our campaign:") 
        print(imperfect_response)
        print("Generating audio/song")
        speech_patterns = say(imperfect_response, voice_name)
        vocalise(speech_patterns)

def clone_from_wav(voice_name, text, audio_filepath):
  model = load_codec_model(use_gpu=True)
  wav, sr = torchaudio.load(audio_filepath)
  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  wav = wav.unsqueeze(0).to(DEVICE)
  model.to(DEVICE)
  with torch.no_grad():
    encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
  model.to(DEVICE)
  seconds = wav.shape[-1] / model.sample_rate
  semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7) # not 100% sure on this part
  codes = codes.cpu().numpy()
  output_path = 'bark/assets/prompts/' + voice_name + '.npz'
  np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

def say(text_prompt, voice_name):
  preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
#    path="models"
  )
  return generate_audio(text_prompt, history_prompt=voice_name)


def tune_model(voice_name, training_text, audio_sample):
    clone_from_wav(voice_name, training_text, audio_sample)

# And then there was life..
sentience()
