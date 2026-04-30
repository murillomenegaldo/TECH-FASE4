from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictFromDataRequest(BaseModel):
    prices: list[float] = Field(
        ...,
        min_length=60,
        description="Historical closing prices in chronological order (minimum 60 values)",
    )
    steps: int = Field(default=1, ge=1, le=30, description="Number of future trading days to forecast")


class PredictFromTickerRequest(BaseModel):
    ticker: str = Field(..., description="Yahoo Finance ticker symbol, e.g. 'PETR4.SA'")
    steps: int = Field(default=1, ge=1, le=30)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class PredictionResponse(BaseModel):
    ticker: Optional[str] = None
    last_known_price: float
    predictions: list[float]
    steps: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    symbol: Optional[str] = None
    seq_len: Optional[int] = None
