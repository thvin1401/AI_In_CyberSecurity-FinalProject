from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path

import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.clean_text import clean_email_text
from src.data.hierarchical_tokenizer import HierarchicalTokenizer
from src.models.han import load_han_model
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger("api")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw email text to classify")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Spam threshold")


class PredictResponse(BaseModel):
    label: str
    spam_probability: float
    ham_probability: float
    threshold: float


class HanPredictor:
    def __init__(
        self,
        weights_path: str | Path,
        processed_dir: str | Path,
    ):
        self.weights_path = Path(weights_path)
        self.processed_dir = Path(processed_dir)

        self.tokenizer: HierarchicalTokenizer | None = None
        self.model = None
        self.max_sentences = CFG.preprocessing.max_sentences
        self.max_words = CFG.preprocessing.max_words

    def _ensure_nltk_data(self) -> None:
        # Docker image pre-downloads these, but local runs may need fallback.
        # Avoid path-based discovery quirks by requesting the packages directly.
        for resource in ("punkt", "punkt_tab"):
            ok = nltk.download(resource, quiet=True)
            if not ok:
                log.warning("Could not confirm NLTK resource '%s' download", resource)

        # Final runtime validation for the exact tokenizer path used in inference.
        try:
            nltk.sent_tokenize("hello world. this is a startup check.")
        except Exception as exc:
            raise RuntimeError(
                "NLTK sentence tokenizer is unavailable; ensure 'punkt' and 'punkt_tab' are installed."
            ) from exc

    def load(self) -> None:
        self._ensure_nltk_data()

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.weights_path}")

        tokenizer_path = self.processed_dir / "tokenizer_vocab.json"
        manifest_path = self.processed_dir / "manifest.json"
        embedding_path = self.processed_dir / "embedding_matrix.npy"

        missing = [
            str(p)
            for p in (tokenizer_path, manifest_path, embedding_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing processed artifacts required for serving: " + ", ".join(missing)
            )

        self.tokenizer = HierarchicalTokenizer.load(tokenizer_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.max_sentences = int(cfg["max_sentences"])
        self.max_words = int(cfg["max_words"])

        embedding_matrix = np.load(embedding_path)
        vocab_size = int(embedding_matrix.shape[0])
        embed_dim = int(embedding_matrix.shape[1])

        self.model = load_han_model(
            weights_path=str(self.weights_path),
            max_sentences=self.max_sentences,
            max_words=self.max_words,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding_matrix=embedding_matrix,
        )
        log.info("HAN predictor ready | checkpoint=%s", self.weights_path)

    def predict(self, raw_text: str, threshold: float = 0.5) -> PredictResponse:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Predictor is not initialized")

        cleaned = clean_email_text(subject="", body=raw_text)
        encoded = self.tokenizer.encode(cleaned)
        batch = np.expand_dims(encoded, axis=0)

        spam_prob = float(self.model.predict(batch, verbose=0).flatten()[0])
        ham_prob = float(1.0 - spam_prob)
        label = "spam" if spam_prob >= threshold else "ham"

        return PredictResponse(
            label=label,
            spam_probability=spam_prob,
            ham_probability=ham_prob,
            threshold=threshold,
        )


def _discover_weights(default_dir: Path) -> Path:
    env_path = os.environ.get("HAN_WEIGHTS_PATH")
    if env_path:
        return Path(env_path)

    candidates = sorted(default_dir.rglob("*.weights.h5"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Set HAN_WEIGHTS_PATH or place *.weights.h5 under checkpoints/."
        )

    selected = candidates[0]
    log.warning("HAN_WEIGHTS_PATH not set; using first checkpoint found: %s", selected)
    return selected


def _processed_dir() -> Path:
    return Path(os.environ.get("HAN_PROCESSED_DIR", CFG.datasets.processed_dir))


predictor: HanPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    checkpoints_dir = Path(os.environ.get("HAN_CHECKPOINT_DIR", "checkpoints"))
    weights_path = _discover_weights(checkpoints_dir)
    processed_dir = _processed_dir()

    predictor = HanPredictor(weights_path=weights_path, processed_dir=processed_dir)
    predictor.load()

    yield


app = FastAPI(
    title="HAN Spam Classifier API",
    version="1.0.0",
    description="Phase 5 serving API for spam/ham prediction using trained HAN checkpoints.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    ready = predictor is not None and predictor.model is not None
    return {"status": "ok" if ready else "initializing"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    return predictor.predict(raw_text=payload.text, threshold=payload.threshold)
