import numpy as np
import pandas as pd
import pytest

from model.preprocessor import (
    create_sequences,
    extract_close,
    fit_scaler,
    preprocess_pipeline,
    prices_to_sequence,
    split_data,
)


def _make_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 20 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({"Close": prices, "Open": prices, "High": prices + 1, "Low": prices - 1})


class TestExtractClose:
    def test_shape(self):
        df = _make_df(100)
        arr = extract_close(df)
        assert arr.shape == (100, 1)

    def test_dtype(self):
        arr = extract_close(_make_df(10))
        assert arr.dtype == float


class TestSplitData:
    def test_lengths(self):
        data = np.arange(200).reshape(-1, 1).astype(float)
        train, val, test = split_data(data, 0.70, 0.15)
        assert len(train) == 140
        assert len(val) == 30
        assert len(test) == 30

    def test_no_overlap(self):
        data = np.arange(100).reshape(-1, 1).astype(float)
        train, val, test = split_data(data)
        assert len(train) + len(val) + len(test) == 100


class TestCreateSequences:
    def test_output_shapes(self):
        data = np.random.rand(100, 1)
        X, y = create_sequences(data, seq_len=10)
        assert X.shape == (90, 10, 1)
        assert y.shape == (90,)

    def test_sequence_values(self):
        data = np.arange(20).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, seq_len=5)
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        assert y[0] == 5.0


class TestPreprocessPipeline:
    def test_pipeline_keys(self):
        df = _make_df(250)
        result = preprocess_pipeline(df, seq_len=30)
        for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "scaler"):
            assert key in result

    def test_pipeline_shapes(self):
        df = _make_df(250)
        result = preprocess_pipeline(df, seq_len=30)
        assert result["X_train"].shape[2] == 1
        assert result["X_train"].shape[1] == 30

    def test_scaled_range(self):
        df = _make_df(250)
        result = preprocess_pipeline(df, seq_len=30)
        assert result["y_train"].min() >= -0.1
        assert result["y_train"].max() <= 1.1


class TestPricesToSequence:
    def test_output_shape(self):
        df = _make_df(200)
        arr = extract_close(df)
        train, _, _ = split_data(arr)
        scaler = fit_scaler(train)
        prices = arr.flatten().tolist()
        seq = prices_to_sequence(prices, scaler, seq_len=60)
        assert seq.shape == (1, 60, 1)
