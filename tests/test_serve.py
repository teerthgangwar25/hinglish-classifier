"""API tests. Most use a fake predictor, so they run without the model (and later in CI).
Tests that need the real model are skipped until `python src/serve/prepare_bundle.py` has been run."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from serve.app import MAX_BATCH, MAX_CHARS, Predictor, create_app  # noqa: E402
from serve.text import clean_text, features, script_class  # noqa: E402


class FakePredictor:
    """Scores 0.9 if the text contains 'bura', 0.5 if it contains 'edge', else 0.1. Threshold 0.5."""
    threshold = 0.5
    meta = {"model_run": "fake", "model_version": "fake-v1", "format": "fake", "threshold": 0.5}

    def predict_proba(self, texts):
        return np.array([0.9 if "bura" in t else 0.5 if "edge" in t else 0.1 for t in texts])


@pytest.fixture()
def client(tmp_path):
    app = create_app(predictor=FakePredictor(), log_db=str(tmp_path / "log.db"), salt="test-salt")
    with TestClient(app) as c:
        c.db = tmp_path / "log.db"
        yield c


# ---------------------------------------------------------------- shape and happy path
def test_health_and_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    v = client.get("/version").json()
    assert v["model_version"] == "fake-v1" and v["threshold"] == 0.5


def test_predict_known_inputs(client):
    bad = client.post("/predict", json={"text": "yeh bahut bura hai"}).json()
    good = client.post("/predict", json={"text": "yeh bahut achha hai"}).json()
    assert bad["label"] == "aggressive" and bad["prob_aggressive"] == 0.9
    assert good["label"] == "not_aggressive"
    assert set(bad) == {"label", "prob_aggressive", "threshold", "model_version", "latency_ms"}


def test_score_equal_to_threshold_counts_as_aggressive(client):
    assert client.post("/predict", json={"text": "edge case"}).json()["label"] == "aggressive"


def test_batch_keeps_order(client):
    r = client.post("/predict_batch", json={"texts": ["bura", "achha", "bura", "achha"]}).json()
    assert [x["label"] for x in r["results"]] == ["aggressive", "not_aggressive", "aggressive", "not_aggressive"]
    assert "latency_ms" in r


# ---------------------------------------------------------------- bad input
@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   \n "}, {"text": 5}, {"text": "a" * (MAX_CHARS + 1)}])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [{}, {"texts": []}, {"texts": ["ok", " "]}, {"texts": ["ok"] * (MAX_BATCH + 1)}])
def test_batch_rejects_bad_input(client, payload):
    assert client.post("/predict_batch", json=payload).status_code == 422


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404


# ---------------------------------------------------------------- logging: hash and features only
def test_log_row_has_hash_and_features_but_no_text(client):
    text = "yeh bahut bura hai http://t.co/abc #IndvsPak"
    client.post("/predict", json={"text": text})
    rows = sqlite3.connect(client.db).execute("SELECT * FROM predictions").fetchall()
    assert len(rows) == 1
    cols = [d[0] for d in sqlite3.connect(client.db).execute("SELECT * FROM predictions").description]
    row = dict(zip(cols, rows[0]))
    assert len(row["text_hash"]) == 16 and row["label"] == 1 and row["has_url"] == 1 and row["has_hashtag"] == 1
    assert row["script"] == "roman" and row["endpoint"] == "/predict" and row["model_version"] == "fake-v1"
    stored = " ".join(str(v) for v in row.values()).lower()
    for piece in ("bahut", "yeh", "indvspak", "t.co"):
        assert piece not in stored            # no part of the comment text is stored


def test_same_text_same_hash_and_batch_rows(client):
    client.post("/predict_batch", json={"texts": ["bura hai", "bura hai", "achha hai"]})
    h = [r[0] for r in sqlite3.connect(client.db).execute("SELECT text_hash FROM predictions ORDER BY id")]
    assert len(h) == 3 and h[0] == h[1] != h[2]
    assert sqlite3.connect(client.db).execute("SELECT DISTINCT batch_size FROM predictions").fetchall() == [(3,)]


def test_logging_can_be_switched_off(tmp_path):
    db = tmp_path / "never.db"
    with TestClient(create_app(predictor=FakePredictor(), log_db="off")) as c:
        assert c.post("/predict", json={"text": "bura"}).status_code == 200
    assert not db.exists()


# ---------------------------------------------------------------- the serving copies of the text helpers
def test_text_helpers_match_the_training_versions():
    pytest.importorskip("pandas")
    from langmix import script_class as ref_script
    from preprocess import clean_text as ref_clean
    samples = ["dekho http://t.co/abc @modi ji sooooo bura", "यह बहुत बुरा है", "yeh बहुत bura", "😂😂 123",
               "bakwas 😂 #IndvsPak", "  spaces   everywhere \t\n ", "tum log पागल ho kya!!!"]
    for s in samples:
        assert clean_text(s) == ref_clean(s)
        assert script_class(clean_text(s)) == ref_script(ref_clean(s))


def test_devanagari_tweet_with_latin_hashtag_is_not_called_mixed():
    f = features("यह बहुत बुरा है #IndvsPak http://t.co/x @modi")
    assert f["script"] == "devanagari" and f["has_hashtag"] == 1 and f["has_mention"] == 1 and f["has_url"] == 1


# ---------------------------------------------------------------- real model (skipped until the bundle exists)
BUNDLE = ROOT / "serving" / "model"
needs_model = pytest.mark.skipif(not (BUNDLE / "meta.json").exists(), reason="run prepare_bundle.py first")


@needs_model
def test_real_model_single_equals_batch(tmp_path):
    texts = ["yeh movie bahut achhi hai", "tum log bilkul pagal ho kya", "बहुत बढ़िया", "bakwas hai ye sab"]
    with TestClient(create_app(log_db="off")) as c:
        single = [c.post("/predict", json={"text": t}).json()["prob_aggressive"] for t in texts]
        batch = [r["prob_aggressive"] for r in c.post("/predict_batch", json={"texts": texts}).json()["results"]]
        assert all(0 <= p <= 1 for p in single)
        assert np.allclose(single, batch, atol=2e-3)       # padding changes the numbers very slightly
        assert c.get("/version").json()["format"] == "onnx-fp32"
