"""
src/data/load_datasets.py

Parses raw datasets into a unified DataFrame: [text, label, source]
  label: 1 = spam, 0 = ham

Expected layout under data/raw/ (you place files here manually):

  data/raw/
    enron_spam_data.csv          <- from MWiechmann/enron_spam_data GitHub zip
    spamassassin/
      easy_ham/                  <- extracted directly, no date prefix
      easy_ham_2/
      hard_ham/
      spam/
      spam_2/

Article ref (Zavrak & Yilmaz 2023, Sec 4.1):
  - EN:  all 6 Enron subsets merged into spam/ham
  - SA:  easy_ham + easy_ham_2 + hard_ham = ham; spam + spam_2 = spam
  - Subject + body extracted; re/fwd/fw tags stripped; HTML removed;
    punctuation removed; lowercased.
"""
import email
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clean_text import clean_email_text
from src.utils.logger import get_logger

log = get_logger("load_datasets")

RAW_DIR = Path("data/raw")
SYNTHETIC_DEFAULT_PATH = RAW_DIR / "synthetic_spam_ham_dataset.csv"

# Tags to strip from subject line (article Sec 4.3 preprocessing)
_REPLY_TAG_RE = re.compile(r"\b(re|fwd|fw)\b[\s:]*", re.IGNORECASE)


def _strip_reply_tags(subject: str) -> str:
    return _REPLY_TAG_RE.sub("", subject).strip()


def _read_file(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def _parse_rfc822(raw: str) -> tuple[str, str]:
    """Parse a raw RFC-822 email into (subject, body)."""
    try:
        msg = email.message_from_string(raw)
        subject = msg.get("Subject", "") or ""
        subject = _strip_reply_tags(subject)
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload.decode("utf-8", errors="ignore"))
            body = "\n".join(parts)
        else:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="ignore") if payload else (msg.get_payload() or "")
        return subject, body
    except Exception:
        return "", raw


# ── Enron-Spam ─────────────────────────────────────────────────────────────────
def load_enron() -> pd.DataFrame:
    """
    Load from the MWiechmann GitHub CSV mirror.
    Columns: Subject, Message, Spam/Ham, Date
    The article merged all 6 Enron subsets into one ham/spam pool — the CSV
    already does this, so we just read it directly.
    """
    csv_path = RAW_DIR / "enron_spam_data.csv"
    if not csv_path.exists():
        log.warning(
            f"Enron CSV not found at {csv_path}. "
            "Download from: https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
        )
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"Loading Enron-Spam from {csv_path} ...")
    df_raw = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")

    records = []
    for _, row in df_raw.iterrows():
        subject = str(row.get("Subject") or "")
        subject = _strip_reply_tags(subject)
        body = str(row.get("Message") or "")
        label_str = str(row.get("Spam/Ham") or "").strip().lower()
        if label_str not in ("spam", "ham"):
            continue
        label = 1 if label_str == "spam" else 0
        records.append({
            "text": clean_email_text(subject, body),
            "label": label,
            "source": "enron",
        })

    df = pd.DataFrame(records)
    log.info(
        f"Loaded Enron: {len(df)} emails "
        f"({(df['label']==1).sum()} spam / {(df['label']==0).sum()} ham)"
    )
    return df


# ── SpamAssassin ───────────────────────────────────────────────────────────────
def load_spamassassin() -> pd.DataFrame:
    """
    Load SpamAssassin corpus from:
        data/raw/spamassassin/easy_ham/     -> ham
        data/raw/spamassassin/easy_ham_2/   -> ham
        data/raw/spamassassin/hard_ham/     -> ham
        data/raw/spamassassin/spam/         -> spam
        data/raw/spamassassin/spam_2/       -> spam

    Folder names match what you get after extracting the .tar.bz2 files
    (no date prefix — the archives unpack to just e.g. easy_ham/).

    Per article Sec 4.1: all ham folders merged as ham, all spam folders as spam.
    """
    sa_dir = RAW_DIR / "spamassassin"
    if not sa_dir.exists():
        log.warning(
            f"SpamAssassin directory not found: {sa_dir}. "
            "Extract the .tar.bz2 archives from https://spamassassin.apache.org/old/publiccorpus/"
        )
        return pd.DataFrame(columns=["text", "label", "source"])

    ham_folders = ["easy_ham", "easy_ham_2", "hard_ham"]
    spam_folders = ["spam", "spam_2"]

    records = []

    for folder_name in ham_folders:
        folder = sa_dir / folder_name
        if not folder.exists():
            log.warning(f"  SpamAssassin folder not found (skipping): {folder}")
            continue
        for fp in folder.iterdir():
            if not fp.is_file() or fp.name.startswith("cmds"):
                continue
            raw = _read_file(fp)
            subject, body = _parse_rfc822(raw)
            records.append({
                "text": clean_email_text(subject, body),
                "label": 0,
                "source": "spamassassin",
            })

    for folder_name in spam_folders:
        folder = sa_dir / folder_name
        if not folder.exists():
            log.warning(f"  SpamAssassin folder not found (skipping): {folder}")
            continue
        for fp in folder.iterdir():
            if not fp.is_file() or fp.name.startswith("cmds"):
                continue
            raw = _read_file(fp)
            subject, body = _parse_rfc822(raw)
            records.append({
                "text": clean_email_text(subject, body),
                "label": 1,
                "source": "spamassassin",
            })

    df = pd.DataFrame(records)
    log.info(
        f"Loaded SpamAssassin: {len(df)} emails "
        f"({(df['label']==1).sum()} spam / {(df['label']==0).sum()} ham)"
    )
    return df


def load_synthetic(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load optional synthetic CSV with schema:
      Subject, Message, Spam/Ham, ...

    Keeps binary label mapping used by the current HAN pipeline:
      spam -> 1, ham -> 0
    """
    path = csv_path or SYNTHETIC_DEFAULT_PATH
    if not path.exists():
        log.warning(f"Synthetic CSV not found at {path}; skipping synthetic data")
        return pd.DataFrame(columns=["text", "label", "source"])

    log.info(f"Loading synthetic dataset from {path} ...")
    df_raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    required = {"Subject", "Message", "Spam/Ham"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            "Synthetic CSV missing required columns: " + ", ".join(sorted(missing))
        )

    records = []
    for _, row in df_raw.iterrows():
        subject = _strip_reply_tags(str(row.get("Subject") or ""))
        body = str(row.get("Message") or "")
        label_str = str(row.get("Spam/Ham") or "").strip().lower()
        if label_str not in ("spam", "ham"):
            continue
        label = 1 if label_str == "spam" else 0
        records.append(
            {
                "text": clean_email_text(subject, body),
                "label": label,
                "source": "synthetic",
            }
        )

    df = pd.DataFrame(records)
    log.info(
        f"Loaded synthetic: {len(df)} emails "
        f"({(df['label']==1).sum()} spam / {(df['label']==0).sum()} ham)"
    )
    return df


# ── Combined ───────────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    """
    Load + merge both datasets.
    Article uses EN and SA separately for cross-dataset evaluation,
    so we keep the 'source' column for that purpose in Phase 4.
    """
    include_synthetic = (
        os.environ.get("HAN_INCLUDE_SYNTHETIC", "false").strip().lower()
        in ("1", "true", "yes", "y")
    )

    df_enron = load_enron()
    df_sa = load_spamassassin()

    parts = [df_enron, df_sa]
    if include_synthetic:
        parts.append(load_synthetic())
        log.info("Synthetic augmentation enabled via HAN_INCLUDE_SYNTHETIC")
    else:
        log.info("Synthetic augmentation disabled (set HAN_INCLUDE_SYNTHETIC=true to enable)")

    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df = df[df["text"].str.strip().str.len() > 10]   # drop near-empty emails
    df = df.drop_duplicates(subset=["text"])
    after = len(df)

    log.info(f"Combined: {after} emails (dropped {before - after} empty/duplicate)")
    log.info(f"  spam: {(df['label']==1).sum()}  ham: {(df['label']==0).sum()}")
    log.info(f"  from enron: {(df['source']=='enron').sum()}  "
             f"from spamassassin: {(df['source']=='spamassassin').sum()}  "
             f"from synthetic: {(df['source']=='synthetic').sum()}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_all()
    out = Path("data/processed/combined_clean.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"Saved -> {out}")
