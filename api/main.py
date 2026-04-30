"""
FastAPI application — Stock Price Prediction API.

Endpoints:
  GET  /health               → liveness/readiness probe
  POST /predict/data         → predict from a list of raw closing prices
  GET  /predict/ticker/{sym} → predict from a Yahoo Finance ticker (live data)
  GET  /metrics              → Prometheus metrics (auto-exposed)
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

from api.predictor import StockPredictor
from api.schemas import (
    HealthResponse,
    PredictFromDataRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Custom Prometheus metrics
INFERENCE_LATENCY = Histogram(
    "stock_inference_latency_seconds",
    "Time spent running LSTM inference",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)
PREDICTION_REQUESTS = Counter(
    "stock_prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

predictor: StockPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model…")
    predictor = StockPredictor()
    if predictor.is_loaded():
        logger.info("Model loaded (symbol=%s, seq_len=%d)", predictor.symbol, predictor.seq_len)
    else:
        logger.warning("Model not found in saved_model/. Run `python train.py` first.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based next-day closing price forecaster.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        symbol=predictor.symbol if predictor else None,
        seq_len=predictor.seq_len if predictor else None,
    )


@app.post("/predict/data", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_data(request: PredictFromDataRequest) -> PredictionResponse:
    """
    Predict closing price from a caller-supplied list of historical prices.

    - **prices**: at least 60 chronological closing prices
    - **steps**: number of trading days to forecast (1–30)
    """
    if not predictor or not predictor.is_loaded():
        PREDICTION_REQUESTS.labels(endpoint="data", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    t0 = time.perf_counter()
    try:
        result = predictor.predict_from_prices(request.prices, request.steps)
        INFERENCE_LATENCY.observe(time.perf_counter() - t0)
        PREDICTION_REQUESTS.labels(endpoint="data", status="ok").inc()
        return PredictionResponse(**result)
    except ValueError as exc:
        PREDICTION_REQUESTS.labels(endpoint="data", status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_REQUESTS.labels(endpoint="data", status="error").inc()
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc


@app.get("/predict/ticker/{ticker}", response_model=PredictionResponse, tags=["Prediction"])
def predict_from_ticker(
    ticker: str,
    steps: int = Query(default=1, ge=1, le=30, description="Days to forecast"),
) -> PredictionResponse:
    """
    Predict closing price by downloading the latest 6 months of data from Yahoo Finance.

    - **ticker**: e.g. `PETR4.SA`, `AAPL`, `MSFT`
    - **steps**: number of trading days to forecast (1–30)
    """
    if not predictor or not predictor.is_loaded():
        PREDICTION_REQUESTS.labels(endpoint="ticker", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    t0 = time.perf_counter()
    try:
        result = predictor.predict_from_ticker(ticker.upper(), steps)
        INFERENCE_LATENCY.observe(time.perf_counter() - t0)
        PREDICTION_REQUESTS.labels(endpoint="ticker", status="ok").inc()
        return PredictionResponse(**result)
    except ValueError as exc:
        PREDICTION_REQUESTS.labels(endpoint="ticker", status="error").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_REQUESTS.labels(endpoint="ticker", status="error").inc()
        logger.exception("Ticker prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc
