#!/usr/bin/env python
"""
src/data/download.py
Download Enron-Spam and SpamAssassin datasets into data/raw/.

Enron-Spam:  http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam
SpamAssassin: https://spamassassin.apache.org/old/publiccorpus/

Usage:
    python src/data/download.py
"""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

log = get_logger("download")

RAW_DIR = Path("data/raw")

# ── Dataset URLs ──────────────────────────────────────────────────────────────
ENRON_URLS = [
    ("enron1",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz"),
    ("enron2",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron2.tar.gz"),
    ("enron3",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron3.tar.gz"),
    ("enron4",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron4.tar.gz"),
    ("enron5",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron5.tar.gz"),
    ("enron6",  "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron6.tar.gz"),
]

SPAMASSASSIN_URLS = [
    ("sa_easy_ham",   "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"),
    ("sa_easy_ham_2", "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"),
    ("sa_hard_ham",   "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2"),
    ("sa_spam",       "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"),
    ("sa_spam_2",     "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2"),
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        log.info(f"  already exists, skipping: {dest.name}")
        return
    log.info(f"  downloading {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        log.error(f"  FAILED {url}: {e}")
        raise


def extract(archive: Path, dest_dir: Path) -> None:
    log.info(f"  extracting {archive.name} → {dest_dir.name}/")
    if archive.suffix == ".gz" or archive.suffixes[-2:] == [".tar", ".gz"]:
        mode = "r:gz"
    else:
        mode = "r:bz2"
    with tarfile.open(archive, mode) as tf:
        tf.extractall(dest_dir)


def download_enron() -> None:
    dest = RAW_DIR / "enron"
    dest.mkdir(parents=True, exist_ok=True)
    log.info("── Enron-Spam ──────────────────────────────")
    for name, url in ENRON_URLS:
        archive = dest / f"{name}.tar.gz"
        download_file(url, archive)
        subset_dir = dest / name
        if not subset_dir.exists():
            extract(archive, dest)
    log.info("Enron-Spam download complete.")


def download_spamassassin() -> None:
    dest = RAW_DIR / "spamassassin"
    dest.mkdir(parents=True, exist_ok=True)
    log.info("── SpamAssassin ────────────────────────────")
    for name, url in SPAMASSASSIN_URLS:
        ext = ".tar.bz2" if url.endswith(".bz2") else ".tar.gz"
        archive = dest / f"{name}{ext}"
        download_file(url, archive)
        subset_dir = dest / name
        if not subset_dir.exists():
            extract(archive, dest)
    log.info("SpamAssassin download complete.")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_enron()
    download_spamassassin()
    log.info("All datasets downloaded to data/raw/")
