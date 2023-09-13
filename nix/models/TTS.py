import os, re
import pickle
import numpy as np
import onnxruntime as ort
import soundfile as sf
from nix.tokenizers.tokenizer_en import NixTokenizerEN

class NixTTSInference:

    def __init__(
        self,
        model_dir,
    ):
        # Load tokenizer
        self.tokenizer = NixTokenizerEN(pickle.load(open(os.path.join(model_dir, "tokenizer_state.pkl"), "rb")))
        # Load TTS model
        self.encoder = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"))
        self.decoder = ort.InferenceSession(os.path.join(model_dir, "decoder.onnx"))

    def tokenize(
        self,
        text,
    ):
        # Tokenize input text
        c, c_lengths, phonemes = self.tokenizer([text])

        return np.array(c, dtype = np.int64), np.array(c_lengths, dtype = np.int64), phonemes

    def vocalize(
        self,
        c,
        c_lengths,
    ):
        """
        Single-batch TTS inference
        """
        # Infer latent samples from encoder
        z = self.encoder.run(
            None,
            {
                "c": c,
                "c_lengths": c_lengths,
            }
        )[2]
        # Decode raw audio with decoder
        xw = self.decoder.run(
            None,
            {
                "z": z,
            }
        )[0]

        return xw

def segment_text(text, max_length):
    """
    Segment input text into smaller parts of approximately maximum length `max_length`
    In order to avoid cutting words in the middle, the text is first split into phrases.
    If the phrase is longer than `max_length`, it is split into smaller phrases.
    """
    segments = []
    phrases = re.findall(r'(?s)(?<=^|(?<=[^a-zA-Z0-9-_.]))(.*?)(?=$|(?=[^a-zA-Z0-9-_.]))', text)
    segment = ""
    for phrase in phrases:
        if len(segment) + len(phrase) + 1 <= max_length:
            segment += phrase + " "
        else:
            segments.append(segment.strip())
            segment = phrase + " "
    if segment:
        segments.append(segment.strip())
    return segments


def generate_audio_segments(nix, segments):
    """
    Generate audio segments for each input text segment
    """
    audio_segments = []
    for segment in segments:
        c, c_length, phoneme = nix.tokenize(segment)
        xw = nix.vocalize(c, c_length)
        audio_segments.append(xw[0,0])
    return audio_segments

def concatenate_audio_segments(audio_segments, sample_rate, filename):
    """
    Concatenate audio segments into a single audio file
    """
    audio = np.concatenate(audio_segments)
    sf.write(filename, audio, sample_rate)

def exportAudio(nix, text, max_length=150, sample_rate=22050, filename="output.wav"):
    """
    Export audio file from input text
    """
    segments = segment_text(text, max_length)
    audio_segments = generate_audio_segments(nix, segments)
    concatenate_audio_segments(audio_segments, sample_rate, filename)
    return filename