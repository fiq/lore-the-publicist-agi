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

This project fixed MPS issue in bark-with-voice-cloning. A script has been provided for voice cloning.

1. Make a mono wav recording (at 24hz sample rate) of yourself saying 'When the previous program was run, you should have been presented with the assembler listing.'
1. Move this to a file called training.wav in the project root folder
1. Run the following:

```

SUNO_ENABLE_MPS=True PYTORCH_ENABLE_MPS_FALLBACK=True python ./clone.py

```


