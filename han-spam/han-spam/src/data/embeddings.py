"""
src/data/embeddings.py
Train a FastText model on the training corpus and build an embedding matrix
aligned with the HierarchicalTokenizer vocabulary (dim=200, paper value).
"""
import sys
import tempfile
from pathlib import Path

import fasttext
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clean_text import sentence_tokenize, word_tokenize
from src.data.hierarchical_tokenizer import HierarchicalTokenizer
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger("embeddings")


def _texts_to_fasttext_corpus(texts: list[str], out_path: Path) -> None:
    """FastText needs a plain text file, one 'document' (space-joined tokens) per line."""
    with open(out_path, "w", encoding="utf-8") as f:
        for text in texts:
            words = []
            for sent in sentence_tokenize(text):
                words.extend(word_tokenize(sent))
            if words:
                f.write(" ".join(words) + "\n")


def train_fasttext(texts: list[str], save_path: str | Path | None = None) -> fasttext.FastText._FastText:
    """Train a FastText skip-gram model on the (training-set only) corpus."""
    save_path = Path(save_path or CFG.fasttext.model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        corpus_path = Path(tmp.name)
    _texts_to_fasttext_corpus(texts, corpus_path)

    log.info(f"Training FastText: dim={CFG.fasttext.dim}, epoch={CFG.fasttext.epoch}, "
             f"thread={getattr(CFG.fasttext, 'thread', 4)}")
    model = fasttext.train_unsupervised(
        str(corpus_path),
        model="skipgram",
        dim=CFG.fasttext.dim,
        epoch=CFG.fasttext.epoch,
        minCount=CFG.fasttext.min_count,
        wordNgrams=CFG.fasttext.word_ngrams,
        thread=getattr(CFG.fasttext, "thread", 4),
    )
    model.save_model(str(save_path))
    corpus_path.unlink(missing_ok=True)
    log.info(f"Saved FastText model → {save_path}")
    return model


def build_embedding_matrix(
    tokenizer: HierarchicalTokenizer,
    fasttext_model: fasttext.FastText._FastText,
) -> np.ndarray:
    """
    Build a [vocab_size, dim] embedding matrix aligned with the tokenizer's word2idx.
    FastText handles OOV words gracefully via sub-word n-grams, so even rare/unseen
    words get a reasonable vector instead of falling back to zeros.
    """
    dim = CFG.fasttext.dim
    vocab_size = tokenizer.vocab_size
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)

    log.info(f"Building embedding matrix: [{vocab_size}, {dim}]")
    for word, idx in tokenizer.word2idx.items():
        if word in ("<PAD>",):
            continue  # stays zero vector
        matrix[idx] = fasttext_model.get_word_vector(word)

    return matrix


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/processed/train_split.parquet")
    texts = df["text"].tolist()

    ft_model = train_fasttext(texts)

    tokenizer = HierarchicalTokenizer.load("data/processed/tokenizer_vocab.json")
    emb_matrix = build_embedding_matrix(tokenizer, ft_model)

    out_path = Path("data/processed/embedding_matrix.npy")
    np.save(out_path, emb_matrix)
    log.info(f"Saved embedding matrix → {out_path}  shape={emb_matrix.shape}")
