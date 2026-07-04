"""
src/data/hierarchical_tokenizer.py
Convert cleaned email text into the hierarchical [sentences, words] integer
tensor structure required by the HAN model, plus a word-index vocabulary.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clean_text import sentence_tokenize, word_tokenize
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger("hierarchical_tokenizer")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class HierarchicalTokenizer:
    """
    Builds a word→index vocabulary from training texts and encodes any text
    into a fixed-size [max_sentences, max_words] integer matrix.
    """

    def __init__(
        self,
        max_sentences: int | None = None,
        max_words: int | None = None,
        min_word_freq: int | None = None,
    ):
        self.max_sentences = max_sentences or CFG.preprocessing.max_sentences
        self.max_words = max_words or CFG.preprocessing.max_words
        self.min_word_freq = min_word_freq or CFG.preprocessing.min_word_freq

        self.word2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._fitted = False

    # ── Vocabulary building ────────────────────────────────────────────────
    def fit(self, texts: list[str]) -> "HierarchicalTokenizer":
        """Build the vocabulary from a list of cleaned texts (training set only!)."""
        counter: Counter = Counter()
        log.info(f"Building vocabulary from {len(texts)} documents ...")
        for text in tqdm(texts, desc="vocab"):
            for sent in sentence_tokenize(text):
                counter.update(word_tokenize(sent))

        idx = 2  # 0=PAD, 1=UNK already reserved
        for word, freq in counter.most_common():
            if freq < self.min_word_freq:
                continue
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        self._fitted = True
        log.info(f"Vocabulary size: {len(self.word2idx)} "
                  f"(min_freq={self.min_word_freq}, raw unique words={len(counter)})")
        return self

    # ── Encoding ────────────────────────────────────────────────────────────
    def encode(self, text: str) -> np.ndarray:
        """
        Encode one cleaned email text into a [max_sentences, max_words] int32 matrix.
        Truncates/pads with 0 (PAD) as needed.
        """
        if not self._fitted:
            raise RuntimeError("Tokenizer not fitted — call .fit() first or .load()")

        matrix = np.zeros((self.max_sentences, self.max_words), dtype=np.int32)
        sentences = sentence_tokenize(text)[: self.max_sentences]

        for i, sent in enumerate(sentences):
            words = word_tokenize(sent)[: self.max_words]
            for j, word in enumerate(words):
                matrix[i, j] = self.word2idx.get(word, self.word2idx[UNK_TOKEN])

        return matrix

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts → [N, max_sentences, max_words] int32 array."""
        out = np.zeros((len(texts), self.max_sentences, self.max_words), dtype=np.int32)
        for i, text in enumerate(tqdm(texts, desc="encode")):
            out[i] = self.encode(text)
        return out

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_sentences": self.max_sentences,
            "max_words": self.max_words,
            "min_word_freq": self.min_word_freq,
            "word2idx": self.word2idx,
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        log.info(f"Saved tokenizer vocabulary → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalTokenizer":
        with open(path, "r") as f:
            payload = json.load(f)
        tok = cls(
            max_sentences=payload["max_sentences"],
            max_words=payload["max_words"],
            min_word_freq=payload["min_word_freq"],
        )
        tok.word2idx = payload["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        tok._fitted = True
        log.info(f"Loaded tokenizer vocabulary from {path} (vocab size={len(tok.word2idx)})")
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
