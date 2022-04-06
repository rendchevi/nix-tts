# **🐤 Nix-TTS**

### **An Incredibly Lightweight End-to-End Text-to-Speech Model via Non End-to-End Distillation**

#### Rendi Chevi, Radityo Eko Prasojo, Alham Fikri Aji

This is a repository for our paper, **🐤 Nix-TTS** (Submitted to INTERSPEECH 2022). We released the pretrained models, an interactive demo, and audio samples below.

[[📄 Paper Link](https://arxiv.org/abs/2203.15643)] [[🤗 Interactive Demo](https://huggingface.co/spaces/rendchevi/nix-tts)] [[📢 Audio Samples](https://drive.google.com/drive/folders/1BJunQY8nBQW5YyZ4MuFN_-T-m91Dk508?usp=sharing)]

**Abstract**&nbsp;&nbsp;&nbsp;&nbsp;We propose Nix-TTS, a lightweight neural TTS (Text-to-Speech) model achieved by applying knowledge distillation to a powerful yet large-sized generative TTS teacher model. Distilling a TTS model might sound unintuitive due to the generative and disjointed nature of TTS architectures, but pre-trained TTS models can be simplified into encoder and decoder structures, where the former encodes text into some latent representation and the latter decodes the latent into speech data. We devise a framework to distill each component in a non end-to-end fashion. Nix-TTS is end-to-end (vocoder-free) with only 5.23M parameters or up to 82% reduction of the teacher model, it achieves over 3.26x and 8.36x inference speedup on Intel-i7 CPU and Raspberry Pi respectively, and still retains a fair voice naturalness and intelligibility compared to the teacher model.

## **Getting Started with Nix-TTS**
**Clone the `nix-tts` repository and move to its directory**
```bash
git clone https://github.com/rendchevi/nix-tts.git
cd nix-tts
```

**Install the dependencies**
- Install Python dependencies. We recommend `python >= 3.8`
```bash
pip install -r requirements.txt 
```
- Install espeak in your device (for text tokenization).
```bash
sudo apt-get install espeak
```
Or follow the [official instruction](https://github.com/bootphon/phonemizer#dependencies) in case it didn't work.

**Download your chosen pre-trained model [here](https://drive.google.com/drive/folders/1GbFOnJsgKHCAXySm2sTluRRikc4TAWxJ?usp=sharing)**. 

| Model      | Num. of Params | Faster than real-time<sup>*</sup> (CPU Intel-i7) | Faster than real-time<sup>*</sup> (RasPi Model 3B) |
| ----------  | -------------- | ----| ----|
| Nix-TTS (ONNX)     | 5.23 M | 11.9x | 0.50x |
| Nix-TTS w/ Stochastic Duration (ONNX) | 6.03 M | 10.8x | 0.50x |

**<sup>*</sup>** Here we compute how much the model run faster than real-time as the inverse of Real Time Factor (RTF). The complete table of all models speedup is detailed on the paper.

**And running Nix-TTS is as easy as:**
```py
from nix.models.TTS import NixTTSInference
from IPython.display import Audio

# Initiate Nix-TTS
nix = NixTTSInference(model_dir = "<path_to_the_downloaded_model>")
# Tokenize input text
c, c_length, phoneme = nix.tokenize("Born to multiply, born to gaze into night skies.")
# Convert text to raw speech
xw = nix.vocalize(c, c_length)

# Listen to the generated speech
Audio(xw[0,0], rate = 22050)
```

## **Acknowledgement**
- This research is fully and exclusively funded by [Kata.ai](https://kata.ai), where the authors work as part of the [Kata.ai Research Team](https://kata.ai/research).
- Some of the complex parts of our model, as mentioned in the paper, are adapted from the original implementation of [VITS](https://github.com/jaywalnut310/vits) and [Comprehensive-Transformer-TTS](https://github.com/keonlee9420/Comprehensive-Transformer-TTS).
