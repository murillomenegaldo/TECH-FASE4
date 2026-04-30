import numpy as np
import pytest


class TestLSTMModel:
    def test_model_builds_without_error(self):
        from model.lstm_model import build_lstm_model
        model = build_lstm_model(seq_len=60)
        assert model is not None

    def test_model_output_shape(self):
        from model.lstm_model import build_lstm_model
        model = build_lstm_model(seq_len=30)
        x = np.random.rand(4, 30, 1).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (4, 1)

    def test_model_has_lstm_layers(self):
        from model.lstm_model import build_lstm_model
        model = build_lstm_model()
        layer_types = [type(l).__name__ for l in model.layers]
        assert "LSTM" in layer_types

    def test_model_compiles(self):
        from model.lstm_model import build_lstm_model
        model = build_lstm_model()
        assert model.optimizer is not None
        assert model.loss is not None
