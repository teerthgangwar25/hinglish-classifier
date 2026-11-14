"""Locust load profile for the API.

Every virtual user sends requests back to back (no think time), so the number of users is the
concurrency level. BATCH_SIZE=1 hits /predict, anything larger hits /predict_batch.
Comments are a fixed random sample of DEV, read from your local data (nothing is written out).
Normally started by run_benchmark.py, not by hand.
"""
import os
import random
import sys
from pathlib import Path

from locust import HttpUser, constant, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import load_frozen_split  # noqa: E402

BATCH = int(os.environ.get("BATCH_SIZE", "1"))
TEXTS = [t[:5000] for t in load_frozen_split("dev")["text"].sample(500, random_state=1).tolist()]


class Client(HttpUser):
    wait_time = constant(0)

    @task
    def predict(self):
        if BATCH == 1:
            self.client.post("/predict", json={"text": random.choice(TEXTS)}, name="/predict")
        else:
            self.client.post("/predict_batch", json={"texts": random.sample(TEXTS, BATCH)},
                             name=f"/predict_batch (x{BATCH})")
