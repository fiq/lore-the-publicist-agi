# LORE - Lippy Orator for Reach and Enterprise

LORE is a mashup of [whisper](https://github.com/openai/whisper) (hearing), [GPT-Neo](https://github.com/EleutherAI/gpt-neo) (generative inference) and bark(generative voicing) + [bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone)

It's core goal is to take a product description and turn it into a jingle, song or marketing approach.

It is a proof of concept for a powerful collection of models which can run FULLY OFFLINE on a local M2 Pro Macbook.
This came joint third in our hackdays. 

* Refer to the source projects for further info. 
* This resulted in an upstream PR for MPS support in Bark. A key insight is that pytorch has a ton of outstanding MPS ops to implement. Great place to contribute if you're bored?
*

# WARNING

~~As this was a hackdays project and I had to make MPS specific tweaks to Bark (specifically [bark with voice clond](https://github.com/serp-ai/bark-with-voice-clone)), this repo contains
a snapshot of the source from this project with MPS fixes. I may return to clean this up, but until such a time please take this as the output of a hackday event.~~

# INSTALLATION

`pip install . --user`

# RUNNING

```

SUNO_ENABLE_MPS=True PYTORCH_ENABLE_MPS_FALLBACK=True python ./lore.py

```

# VOICE TRAINING

This project fixed MPS issue in bark-with-voice-cloning. A script has been provided for voice cloning. Use [bark with voice clone](https://github.com/serp-ai/bark-with-voice-clone) to clone a new voice. 

1. Make a mono wav recording (at 24hz sample rate) of yourself saying, 'Jabba the hut and the evoks like to dance.'
1. Move this to a file called training.wav in the project root folder
1. A script of the following form (TODO - commit this somewhere, can be used to generate a new voice):

```

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
pp = pprint.PrettyPrinter()
preload_models(
  text_use_gpu=True,
  text_use_small=False,
  coarse_use_gpu=True,
  coarse_use_small=False,
  fine_use_gpu=True,
  fine_use_small=False,
  force_reload=False,
  path = "models"
)

#DEVICE = "mps"
DEVICE = "cuda"


def clone_from_wav(voice_name, text, audio_filepath):
  model = load_codec_model(use_gpu=True)
  wav, sr = torchaudio.load(audio_filepath)
  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  av = wav.unsqueeze(0).to(DEVICE)
  wav = wav.unsqueeze(0)
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

# The text you'll say
training_text = """
Hello there. Jabba the hut and the evoks like to dance.
"""
voice_name="lore_clone_0"
audio_sample = "training.wav"
clone_from_wav(voice_name, training_text, audio_sample)

```

To run on a silicon device:

```
SUNO_ENABLE_MPS=True PYTORCH_ENABLE_MPS_FALLBACK=True python ./clone.py
```

