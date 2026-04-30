import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

SEQUENCE_LENGTH = 60


def extract_close(df: pd.DataFrame) -> np.ndarray:
    return df[["Close"]].values.astype(float)


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def fit_scaler(train_data: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    return scaler


def create_sequences(
    scaled_data: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) pairs where X is a window of seq_len prices and y is the next price."""
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len : i, 0])
        y.append(scaled_data[i, 0])
    X_arr = np.array(X).reshape(-1, seq_len, 1)
    y_arr = np.array(y)
    return X_arr, y_arr


def preprocess_pipeline(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """End-to-end preprocessing: split → scale → sequence."""
    raw = extract_close(df)
    train_raw, val_raw, test_raw = split_data(raw, train_ratio, val_ratio)

    scaler = fit_scaler(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # Sequences need the tail of the previous split to avoid losing context
    X_train, y_train = create_sequences(train_scaled, seq_len)

    val_combined = np.concatenate([train_scaled[-seq_len:], val_scaled])
    X_val, y_val = create_sequences(val_combined, seq_len)

    test_combined = np.concatenate([val_scaled[-seq_len:], test_scaled])
    X_test, y_test = create_sequences(test_combined, seq_len)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "raw_test": test_raw,
    }


def prices_to_sequence(prices: list[float], scaler: MinMaxScaler, seq_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Convert a list of raw prices into a model-ready sequence."""
    arr = np.array(prices[-seq_len:], dtype=float).reshape(-1, 1)
    scaled = scaler.transform(arr)
    return scaled.reshape(1, seq_len, 1)
