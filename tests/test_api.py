"""API tests using a tiny model trained on synthetic data."""

import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from src import serve
from src.config import MODEL_FEATURES
from src.features import add_features, add_target, clean_candles
from src.sample_data import make_candles


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = add_target(add_features(clean_candles(make_candles(800, seed=3))))
    df = df.dropna(subset=["target"])
    df["stock_name"] = "DEMO"
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["stock_encoded"] = enc.fit_transform(df[["stock_name"]]).ravel()
    model = HistGradientBoostingClassifier(max_iter=20).fit(df[MODEL_FEATURES], df["target"].astype(int))

    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"model": model, "encoder": enc, "features": MODEL_FEATURES, "threshold": 0.5}, path)

    os.environ["MODEL_PATH"] = str(path)
    serve.load_bundle.cache_clear()
    yield TestClient(serve.app)
    os.environ.pop("MODEL_PATH")
    serve.load_bundle.cache_clear()


def payload(n=15):
    c = make_candles(n, seed=9)
    return {
        "stock_name": "DEMO",
        "candles": [
            {"timestamp": t.isoformat(), "close": p, "volume": v}
            for t, p, v in zip(c.timestamp, c.close, c.volume, strict=True)
        ],
    }


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_up"] <= 1
    assert body["prediction"] in (0, 1)


def test_predict_unknown_stock_still_works(client):
    r = client.post("/predict", json={**payload(), "stock_name": "NEWCO"})
    assert r.status_code == 200


def test_predict_rejects_too_few_candles(client):
    assert client.post("/predict", json=payload(5)).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["known_stocks"] == ["DEMO"]
    assert body["threshold"] == 0.5


def test_metrics_exposed(client):
    client.post("/predict", json=payload())
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert "http_request_latency_seconds" in text
