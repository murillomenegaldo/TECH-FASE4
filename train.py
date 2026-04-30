"""
Main training script.

Usage:
    python train.py [--symbol PETR4.SA] [--start 2018-01-01] [--end 2024-12-31]
                    [--epochs 100] [--batch-size 32] [--seq-len 60]
"""

import argparse
import json
import os

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras

from model.data_collector import download_stock_data
from model.lstm_model import build_lstm_model
from model.preprocessor import preprocess_pipeline

MODEL_DIR = "saved_model"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape_val = mape(actual, predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape_val, 4), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_history(history, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train MAE")
    axes[1].plot(history.history["val_mae"], label="val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Training history plot saved to {path}")


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual, label="Actual", linewidth=1.5)
    ax.plot(predicted, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (BRL)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "predictions_vs_actual.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Prediction plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM stock price predictor")
    p.add_argument("--symbol", default="PETR4.SA", help="Yahoo Finance ticker")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--out-dir", default=MODEL_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data collection
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Stock: {args.symbol}  |  {args.start} → {args.end}")
    print(f"{'='*60}\n")

    df = download_stock_data(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        save_path=os.path.join("data", f"{args.symbol.replace('.', '_')}.csv"),
    )

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    data = preprocess_pipeline(df, seq_len=args.seq_len)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    scaler = data["scaler"]

    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = build_lstm_model(seq_len=args.seq_len)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
        ),
    ]

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    print("\nTraining…")
    history = model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    pred_scaled = model.predict(X_test, verbose=0)
    pred_prices = scaler.inverse_transform(pred_scaled).flatten()
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = compute_metrics(actual_prices, pred_prices)
    print("\n--- Test Set Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    model_path = os.path.join(args.out_dir, "lstm_model.keras")
    model.save(model_path)
    print(f"\nModel saved → {model_path}")

    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved → {scaler_path}")

    meta = {
        "symbol": args.symbol,
        "start_date": args.start,
        "end_date": args.end,
        "seq_len": args.seq_len,
        "metrics": metrics,
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {meta_path}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    plot_training_history(history, args.out_dir)
    plot_predictions(actual_prices, pred_prices, args.out_dir)

    print(f"\nAll artefacts written to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
