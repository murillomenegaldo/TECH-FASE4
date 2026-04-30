import json
import os
from typing import Optional

import joblib
import numpy as np
import yfinance as yf

MODEL_DIR = os.getenv("MODEL_PATH", "saved_model")
_DEFAULT_SEQ_LEN = 60


class StockPredictor:
    """Loads the trained LSTM model and scaler and exposes prediction methods."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.seq_len: int = _DEFAULT_SEQ_LEN
        self.symbol: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        model_path = os.path.join(self.model_dir, "lstm_model.keras")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        meta_path = os.path.join(self.model_dir, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return  # model not yet trained; handled gracefully by is_loaded()

        import tensorflow as tf  # deferred import so the module can be imported without TF

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.seq_len = meta.get("seq_len", _DEFAULT_SEQ_LEN)
            self.symbol = meta.get("symbol")

    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _scale(self, prices: list[float]) -> np.ndarray:
        arr = np.array(prices[-self.seq_len :], dtype=float).reshape(-1, 1)
        return self.scaler.transform(arr)

    def _predict_steps(self, prices: list[float], steps: int) -> list[float]:
        """Recursive multi-step forecasting."""
        window = list(prices[-self.seq_len :])
        results: list[float] = []

        for _ in range(steps):
            scaled_window = self._scale(window)
            x = scaled_window.reshape(1, self.seq_len, 1)
            pred_scaled = self.model.predict(x, verbose=0)
            pred_price = float(self.scaler.inverse_transform(pred_scaled).flatten()[0])
            results.append(round(pred_price, 4))
            window.append(pred_price)
            window = window[-self.seq_len :]

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_prices(self, prices: list[float], steps: int = 1) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Run train.py first.")
        if len(prices) < self.seq_len:
            raise ValueError(f"At least {self.seq_len} prices are required.")
        predictions = self._predict_steps(prices, steps)
        return {
            "last_known_price": round(prices[-1], 4),
            "predictions": predictions,
            "steps": steps,
        }

    def predict_from_ticker(self, ticker: str, steps: int = 1) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Run train.py first.")

        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        prices = hist["Close"].tolist()
        if len(prices) < self.seq_len:
            raise ValueError(f"Not enough history for '{ticker}' (need ≥{self.seq_len} trading days).")

        result = self._predict_steps(prices, steps)
        return {
            "ticker": ticker,
            "last_known_price": round(prices[-1], 4),
            "predictions": result,
            "steps": steps,
        }
