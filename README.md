# **🐤 Nix-TTS**

### **Lightweight and End-to-end Text-to-Speech via Module-wise Distillation**

#### Rendi Chevi, Radityo Eko Prasojo, Alham Fikri Aji, Andros Tjandra, Sakriani Sakti

This is a repository for our paper, **🐤 Nix-TTS** (Accepted to IEEE SLT 2022). We released the pretrained models, an interactive demo, and audio samples below.

[[📄 Paper Link](Coming Soon!)] [[🤗 Interactive Demo](https://huggingface.co/spaces/rendchevi/nix-tts)] [[📢 Audio Samples](https://anon1178.github.io/Nix-SLT-Demo/)]

**Abstract**&nbsp;&nbsp;&nbsp;&nbsp;Several solutions for lightweight TTS have shown promising results. Still, they either rely on a hand-crafted design that reaches non-optimum size or use a neural architecture search but often suffer training costs. We present Nix-TTS, a lightweight TTS achieved via knowledge distillation to a high-quality yet large-sized, non-autoregressive, and end-to-end (vocoder-free) TTS teacher model. Specifically, we offer module-wise distillation, enabling flexible and independent distillation to the encoder and decoder module. The resulting Nix-TTS inherited the advantageous properties of being non-autoregressive and end-to-end from the teacher, yet significantly smaller in size, with only 5.23M parameters or up to 89.34\% reduction of the teacher model; it also achieves over 3.04$\times$ and 8.36$\times$ inference speedup on Intel-i7 CPU and Raspberry Pi 3B respectively and still retains a fair voice naturalness and intelligibility compared to the teacher model.

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
