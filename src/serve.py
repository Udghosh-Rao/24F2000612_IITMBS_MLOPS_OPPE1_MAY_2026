"""FastAPI inference service.

Clients send the most recent minute candles for one stock; the service builds
features with the *same* code used in training (src/features.py) and returns the
probability that the price will be higher 5 minutes after the last candle.

Run:  uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, WINDOW
from src.features import add_features, clean_candles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("stock-api")

app = FastAPI(title="Stock Movement Predictor", version="2.0.0")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
PROBABILITY = Histogram("prediction_probability_up", "Predicted P(up)",
                        buckets=[i / 10 for i in range(11)])


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    log.info("%s %s %s %.1fms", request.method, path, response.status_code, elapsed * 1000)
    return response


class Candle(BaseModel):
    timestamp: str
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class PredictRequest(BaseModel):
    stock_name: str
    candles: list[Candle] = Field(min_length=WINDOW)


class PredictResponse(BaseModel):
    stock_name: str
    as_of: str
    probability_up: float
    prediction: int
    threshold: float


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(os.environ.get("MODEL_PATH", MODEL_PATH))


@app.get("/health")
def health():
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "features": bundle["features"]}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    try:
        bundle = load_bundle()
    except FileNotFoundError as err:
        raise HTTPException(503, "Model artifact not available") from err
    return {
        "model_type": type(bundle["model"]).__name__,
        "features": bundle["features"],
        "threshold": float(bundle.get("threshold", 0.5)),
        "known_stocks": [str(c) for c in bundle["encoder"].categories_[0]],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        bundle = load_bundle()
    except FileNotFoundError as err:
        raise HTTPException(503, "Model artifact not available") from err

    df = pd.DataFrame([c.model_dump() for c in req.candles])
    try:
        df = add_features(clean_candles(df))
    except (ValueError, TypeError) as err:
        raise HTTPException(422, str(err)) from err

    df["stock_encoded"] = bundle["encoder"].transform(
        pd.DataFrame({"stock_name": [req.stock_name] * len(df)})
    ).ravel()

    last = df.iloc[[-1]][bundle["features"]]
    proba = float(bundle["model"].predict_proba(last)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    label = int(proba >= threshold)
    PREDICTIONS.labels("up" if label else "not_up").inc()
    PROBABILITY.observe(proba)

    return PredictResponse(
        stock_name=req.stock_name,
        as_of=df["timestamp"].iloc[-1].isoformat(),
        probability_up=proba,
        prediction=label,
        threshold=threshold,
    )
