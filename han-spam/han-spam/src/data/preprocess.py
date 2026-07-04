#!/usr/bin/env python
"""
src/data/preprocess.py
Phase 2 entrypoint.

Follows the exact protocol from Zavrak & Yilmaz (2023) Sec 4.3:

  1. Load Enron-Spam (CSV) + SpamAssassin (raw RFC-822 files)
  2. Clean text: strip HTML, re/fwd/fw tags, punctuation, lowercase
  3. Keep 'source' column (enron / spamassassin) for cross-dataset eval
  4. Stratified split per dataset:
       - Enron: 10-fold CV split (train/test done inside Colab per fold)
       - SpamAssassin: same
       NOTE: vocabulary is built per-fold inside Colab to prevent leakage,
             exactly as the article describes. Here we only build a GLOBAL
             vocab + FastText on the full training portion for the embedding
             matrix — Colab will re-tokenize per fold using this vocab.
  5. Encode all texts -> [N, max_sentences, max_words] int32 tensors
  6. Train FastText on Enron training portion (per article: FT trained on
     the same dataset being evaluated, not the combined corpus)
  7. Save artefacts to data/processed/ -> push to Drive -> Colab trains

Usage:
    python src/data/preprocess.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.embeddings import build_embedding_matrix, train_fasttext
from src.data.hierarchical_tokenizer import HierarchicalTokenizer
from src.data.load_datasets import load_all
from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger("preprocess")
PROCESSED_DIR = Path(CFG.datasets.processed_dir)


def main():
    set_seed(CFG.project.seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + clean ────────────────────────────────────────────────────
    log.info("STEP 1/5 - loading and cleaning datasets")
    df = load_all()

    if len(df) == 0:
        log.error("No data loaded. Place datasets in data/raw/ and retry.")
        log.error("  data/raw/enron_spam_data.csv")
        log.error("  data/raw/spamassassin/{easy_ham,easy_ham_2,hard_ham,spam,spam_2}/")
        sys.exit(1)

    # Save full combined for reference
    df.to_parquet(PROCESSED_DIR / "combined_clean.parquet", index=False)
    log.info(f"  total emails: {len(df)}")

    # ── 2. Split by source for cross-dataset eval (article protocol) ───────
    log.info("STEP 2/5 - splitting by source + stratified train/test")
    df_enron = df[df["source"] == "enron"].reset_index(drop=True)
    df_sa    = df[df["source"] == "spamassassin"].reset_index(drop=True)

    # 80/20 stratified split per dataset
    en_train, en_test = train_test_split(
        df_enron, test_size=0.2, stratify=df_enron["label"],
        random_state=CFG.project.seed
    )
    sa_train, sa_test = train_test_split(
        df_sa, test_size=0.2, stratify=df_sa["label"],
        random_state=CFG.project.seed
    )

    en_train = en_train.reset_index(drop=True)
    en_test  = en_test.reset_index(drop=True)
    sa_train = sa_train.reset_index(drop=True)
    sa_test  = sa_test.reset_index(drop=True)

    log.info(f"  Enron  train: {len(en_train)}  test: {len(en_test)}")
    log.info(f"  SA     train: {len(sa_train)}  test: {len(sa_test)}")

    # Save splits (Colab uses these for 10-fold CV on each dataset)
    en_train.to_parquet(PROCESSED_DIR / "enron_train.parquet",  index=False)
    en_test.to_parquet( PROCESSED_DIR / "enron_test.parquet",   index=False)
    sa_train.to_parquet(PROCESSED_DIR / "sa_train.parquet",     index=False)
    sa_test.to_parquet( PROCESSED_DIR / "sa_test.parquet",      index=False)

    # ── 3. Fit vocabulary on Enron train only (article trains FT on EN) ────
    log.info("STEP 3/5 - fitting vocabulary (Enron train split only)")
    tokenizer = HierarchicalTokenizer()
    tokenizer.fit(en_train["text"].tolist())
    tokenizer.save(PROCESSED_DIR / "tokenizer_vocab.json")

    # ── 4. Encode all splits with this vocab ───────────────────────────────
    log.info("STEP 4/5 - encoding all splits to hierarchical tensors")
    for name, split_df in [
        ("enron_train", en_train), ("enron_test", en_test),
        ("sa_train",    sa_train), ("sa_test",    sa_test),
    ]:
        X = tokenizer.encode_batch(split_df["text"].tolist())
        y = split_df["label"].to_numpy(dtype=np.int32)
        np.save(PROCESSED_DIR / f"X_{name}.npy", X)
        np.save(PROCESSED_DIR / f"y_{name}.npy", y)
        log.info(f"  {name}: X={X.shape} y={y.shape}")

    # ── 5. FastText on Enron train, build embedding matrix ────────────────
    log.info("STEP 5/5 - training FastText (Enron train only) + embedding matrix")
    ft_model = train_fasttext(en_train["text"].tolist())
    emb_matrix = build_embedding_matrix(tokenizer, ft_model)
    np.save(PROCESSED_DIR / "embedding_matrix.npy", emb_matrix)
    log.info(f"  embedding_matrix: {emb_matrix.shape}")

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "vocab_size":      tokenizer.vocab_size,
        "embedding_dim":   CFG.fasttext.dim,
        "max_sentences":   CFG.preprocessing.max_sentences,
        "max_words":       CFG.preprocessing.max_words,
        "enron_train":     len(en_train),
        "enron_test":      len(en_test),
        "sa_train":        len(sa_train),
        "sa_test":         len(sa_test),
        "enron_spam_ratio": float((en_train["label"] == 1).mean()),
        "sa_spam_ratio":    float((sa_train["label"] == 1).mean()),
    }
    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("-" * 60)
    log.info("Phase 2 complete. Files in data/processed/:")
    for fp in sorted(PROCESSED_DIR.iterdir()):
        log.info(f"  {fp.name:<32} {fp.stat().st_size/1e6:7.2f} MB")
    log.info("-" * 60)
    log.info("Next step:")
    log.info("  python scripts/drive_sync.py push-data")


if __name__ == "__main__":
    main()
