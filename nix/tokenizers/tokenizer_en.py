# Regex
import re

# Phonemizer
from phonemizer.backend import EspeakBackend
phonemizer_backend = EspeakBackend(
    language = 'en-us',
    preserve_punctuation = True,
    with_stress = True
)

class NixTokenizerEN:

    def __init__(
        self,
        tokenizer_state,
    ):
        # Vocab and abbreviations dictionary
        self.vocab_dict = tokenizer_state["vocab_dict"]
        self.abbreviations_dict = tokenizer_state["abbreviations_dict"]

        # Regex recipe
        self.whitespace_regex = tokenizer_state["whitespace_regex"]
        self.abbreviations_regex = tokenizer_state["abbreviations_regex"]

    def __call__(
        self,
        texts,
    ):
        # 1. Phonemize input texts
        phonemes = [ self._collapse_whitespace(
            phonemizer_backend.phonemize(
                self._expand_abbreviations(text.lower()),
                strip = True,
            )
        ) for text in texts ]

        # 2. Tokenize phonemes
        tokens = [ self._intersperse([self.vocab_dict[p] for p in phoneme], 0) for phoneme in phonemes ]

        # 3. Pad tokens
        tokens, tokens_lengths = self._pad_tokens(tokens)

        return tokens, tokens_lengths, phonemes

    def _expand_abbreviations(
        self,
        text
    ):
        for regex, replacement in self.abbreviations_regex:
            text = re.sub(regex, replacement, text)

        return text

    def _collapse_whitespace(
        self,
        text
    ):
        return re.sub(self.whitespace_regex, ' ', text)

    def _intersperse(
        self,
        lst,
        item,
    ):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    def _pad_tokens(
        self,
        tokens,
    ):
        tokens_lengths = [len(token) for token in tokens]
        max_len = max(tokens_lengths)
        tokens = [token + [0 for _ in range(max_len - len(token))] for token in tokens]
        return tokens, tokens_lengths