import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    seq_len: int = 60,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Two-layer stacked LSTM for univariate time-series regression.

    Architecture:
        Input → LSTM(128, return_sequences=True) → Dropout
              → LSTM(64) → Dropout
              → Dense(32, relu) → Dense(1)
    """
    inputs = keras.Input(shape=(seq_len, 1), name="price_sequence")

    x = layers.LSTM(lstm_units_1, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs, output, name="LSTM_StockPredictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
