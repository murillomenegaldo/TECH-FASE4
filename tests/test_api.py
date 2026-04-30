"""
API tests — model does not need to be trained for these to pass.
We mock the predictor to avoid TF/model dependency.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def _make_client():
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        client = _make_client()
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body


class TestPredictFromData:
    def test_missing_model_returns_503(self):
        client = _make_client()
        payload = {"prices": [float(i) for i in range(60)], "steps": 1}
        resp = client.post("/predict/data", json=payload)
        # 503 when model is not loaded; 200 when it is
        assert resp.status_code in (200, 503)

    def test_too_few_prices_returns_422(self):
        client = _make_client()
        payload = {"prices": [1.0] * 10, "steps": 1}
        resp = client.post("/predict/data", json=payload)
        assert resp.status_code == 422

    def test_steps_out_of_range_returns_422(self):
        client = _make_client()
        payload = {"prices": [1.0] * 60, "steps": 100}
        resp = client.post("/predict/data", json=payload)
        assert resp.status_code == 422

    def test_predict_with_mock_model(self):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict_from_prices.return_value = {
            "last_known_price": 30.5,
            "predictions": [31.0],
            "steps": 1,
        }

        with patch("api.main.predictor", mock_predictor):
            client = _make_client()
            payload = {"prices": [float(i) for i in range(1, 61)], "steps": 1}
            resp = client.post("/predict/data", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 1


class TestPredictFromTicker:
    def test_predict_ticker_with_mock(self):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded.return_value = True
        mock_predictor.predict_from_ticker.return_value = {
            "ticker": "PETR4.SA",
            "last_known_price": 38.5,
            "predictions": [39.1, 39.4],
            "steps": 2,
        }

        with patch("api.main.predictor", mock_predictor):
            client = _make_client()
            resp = client.get("/predict/ticker/PETR4.SA?steps=2")

        assert resp.status_code == 200
        body = resp.json()
        assert body["steps"] == 2
        assert len(body["predictions"]) == 2
