import os
import pickle
import timeit

import numpy as np
import onnxruntime as ort

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
