"""FastAPI service for the Hinglish aggression classifier (ONNX fp32, CPU).

Run from the repo root:
    uvicorn serve.app:app --app-dir src --port 8000
Docs and a try-it page: http://127.0.0.1:8000/docs

Endpoints: POST /predict, POST /predict_batch, GET /health, GET /version.
Every prediction is logged to SQLite as a salted hash plus cheap features. The comment text is never stored.
Settings (environment variables): MODEL_DIR, LOG_DB ('off' disables logging), LOG_SALT, ORT_THREADS.
"""
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from serve.text import clean_text, features

ROOT = Path(__file__).resolve().parents[2]
MAX_CHARS = 5000        # the model reads at most 128 tokens anyway; this only stops absurd payloads
MAX_BATCH = 64
log = logging.getLogger("serve")


# ---------------------------------------------------------------- model
class Predictor:
    """Loads the ONNX model and tokenizer ONCE (at startup), then answers many requests."""

    def __init__(self, bundle: Path, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
        self.threshold = float(self.meta["threshold"])
        self.tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        self.tok.enable_truncation(max_length=int(self.meta["max_len"]))
        pad_id = self.tok.token_to_id("[PAD]")
        if pad_id is None:
            raise RuntimeError("tokenizer has no [PAD] token")
        self.tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")      # pad to the longest text in the batch
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(bundle / "model.onnx"), opts, providers=["CPUExecutionProvider"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(aggressive) for each (already cleaned) text."""
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        logits = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True))[:, 1]


# ---------------------------------------------------------------- request / response shapes
def _check_text(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("text must not be empty")
    if len(v) > MAX_CHARS:
        raise ValueError(f"text is longer than {MAX_CHARS} characters")
    return v


class PredictIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _v(cls, v):
        return _check_text(v)


class BatchIn(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def _v(cls, v):
        if not 1 <= len(v) <= MAX_BATCH:
            raise ValueError(f"send between 1 and {MAX_BATCH} texts")
        return [_check_text(t) for t in v]


class PredictOut(BaseModel):
    label: str                 # "aggressive" or "not_aggressive"
    prob_aggressive: float
    threshold: float
    model_version: str


class PredictResponse(PredictOut):
    latency_ms: float


class BatchResponse(BaseModel):
    results: list[PredictOut]
    latency_ms: float


# ---------------------------------------------------------------- logging (SQLite, no raw text)
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL, endpoint TEXT NOT NULL, batch_size INTEGER NOT NULL,
    text_hash TEXT NOT NULL, n_chars INTEGER, n_words INTEGER, script TEXT, deva_share REAL,
    has_hashtag INTEGER, has_mention INTEGER, has_url INTEGER, has_emoji INTEGER,
    label INTEGER NOT NULL, score REAL NOT NULL, threshold REAL NOT NULL,
    model_version TEXT NOT NULL, latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts_utc);
"""
_db_lock = threading.Lock()


def init_db(path: str) -> None:
    if path.lower() == "off":
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)


def write_log(path: str, rows: list[tuple]) -> None:
    """Runs AFTER the response is sent. A logging failure must never break a prediction."""
    if path.lower() == "off":
        return
    try:
        with _db_lock, sqlite3.connect(path, timeout=5) as con:
            con.executemany("INSERT INTO predictions (ts_utc, endpoint, batch_size, text_hash, n_chars, n_words, script, "
                            "deva_share, has_hashtag, has_mention, has_url, has_emoji, label, score, threshold, "
                            "model_version, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    except Exception:
        log.exception("could not write prediction log")


# ---------------------------------------------------------------- the app
def create_app(predictor=None, log_db: str | None = None, salt: str | None = None) -> FastAPI:
    log_db = log_db if log_db is not None else os.environ.get("LOG_DB", str(ROOT / "logs" / "predictions.db"))
    salt = salt if salt is not None else os.environ.get("LOG_SALT", "change-me")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is None:
            bundle = Path(os.environ.get("MODEL_DIR", ROOT / "serving" / "model"))
            if not (bundle / "meta.json").exists():
                raise RuntimeError(f"no model bundle at {bundle}. Run: python src/serve/prepare_bundle.py")
            app.state.predictor = Predictor(bundle, threads=int(os.environ.get("ORT_THREADS", "4")))
        else:
            app.state.predictor = predictor
        init_db(log_db)
        yield

    app = FastAPI(title="Hinglish aggression classifier", version="0.1.0", lifespan=lifespan)

    def _score(texts: list[str]):
        p = app.state.predictor
        t0 = time.perf_counter()
        probs = p.predict_proba([clean_text(t) for t in texts])
        return probs, (time.perf_counter() - t0) * 1000, p

    def _out(prob: float, p) -> PredictOut:
        return PredictOut(label="aggressive" if prob >= p.threshold else "not_aggressive",
                          prob_aggressive=round(float(prob), 4), threshold=p.threshold,
                          model_version=p.meta["model_version"])

    def _rows(endpoint, texts, probs, p, latency):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for t, pr in zip(texts, probs):
            f = features(t)
            digest = hashlib.sha256((salt + clean_text(t)).encode("utf-8")).hexdigest()[:16]
            rows.append((ts, endpoint, len(texts), digest, f["n_chars"], f["n_words"], f["script"], f["deva_share"],
                         f["has_hashtag"], f["has_mention"], f["has_url"], f["has_emoji"],
                         int(pr >= p.threshold), float(pr), p.threshold, p.meta["model_version"], latency))
        return rows

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictIn, bg: BackgroundTasks):
        probs, ms, p = _score([body.text])
        bg.add_task(write_log, log_db, _rows("/predict", [body.text], probs, p, ms))
        return PredictResponse(**_out(probs[0], p).model_dump(), latency_ms=round(ms, 2))

    @app.post("/predict_batch", response_model=BatchResponse)
    def predict_batch(body: BatchIn, bg: BackgroundTasks):
        probs, ms, p = _score(body.texts)
        bg.add_task(write_log, log_db, _rows("/predict_batch", body.texts, probs, p, ms))
        return BatchResponse(results=[_out(x, p) for x in probs], latency_ms=round(ms, 2))

    @app.get("/health")
    def health():
        if getattr(app.state, "predictor", None) is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return app.state.predictor.meta

    return app


app = create_app()
