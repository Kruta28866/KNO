#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras_tuner as kt


def add_fourier_features(df, t, period, harmonics):
    for k in range(1, harmonics + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def make_sequences(df, target_col, window, horizon=1):
    values = df.values
    target_idx = df.columns.get_loc(target_col)
    X, y = [], []
    for i in range(window, len(df) - horizon + 1):
        X.append(values[i - window:i, :])
        y.append(values[i + horizon - 1, target_idx])
    return np.array(X), np.array(y)


def build_rnn(hp, n_features, rnn_type):
    units = hp.Int("units", 16, 128, step=16)
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
    dense_units = hp.Int("dense_units", 8, 64, step=8)
    lr = hp.Choice("lr", [1e-2, 1e-3, 3e-4])

    inp = keras.Input(shape=(None, n_features))
    if rnn_type == "lstm":
        x = layers.LSTM(units, dropout=dropout)(inp)
    else:
        x = layers.GRU(units, dropout=dropout)(inp)

    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    return model


def build_mlp(hp, window, n_features):
    units = hp.Int("units", 32, 256, step=32)
    dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
    dense2 = hp.Int("dense2", 16, 128, step=16)
    lr = hp.Choice("lr", [1e-2, 1e-3, 3e-4])

    inp = keras.Input(shape=(window, n_features))
    x = layers.Flatten()(inp)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense2, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--value_col", default="Close")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--period", type=int, default=7)
    ap.add_argument("--harmonics", type=int, default=2)
    ap.add_argument("--use_diff", action="store_true")
    ap.add_argument("--model", choices=["gru", "lstm", "mlp"], default="gru")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.history)
    series = df[[args.value_col]].copy()

    if args.use_diff:
        series["value"] = series[args.value_col].diff()
        series = series.drop(columns=[args.value_col]).dropna().reset_index(drop=True)
        target_col = "value"
    else:
        series = series.rename(columns={args.value_col: "value"})
        target_col = "value"

    t = np.arange(len(series))
    feat_df = add_fourier_features(series.copy(), t, args.period, args.harmonics)

    scaler = StandardScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(feat_df),
        columns=feat_df.columns
    )

    X, y = make_sequences(feat_scaled, target_col, args.window, args.horizon)

    split = int((1 - args.test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_features = X.shape[-1]

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    if args.tune:
        if args.model in ["gru", "lstm"]:
            tuner = kt.RandomSearch(
                lambda hp: build_rnn(hp, n_features, args.model),
                objective="val_loss",
                max_trials=args.trials,
                directory=str(outdir / "tuner"),
                project_name=args.model
            )
        else:
            tuner = kt.RandomSearch(
                lambda hp: build_mlp(hp, args.window, n_features),
                objective="val_loss",
                max_trials=args.trials,
                directory=str(outdir / "tuner"),
                project_name="mlp"
            )

        tuner.search(
            X_train, y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch,
            callbacks=callbacks
        )
        model = tuner.get_best_models(1)[0]
        best_hp = tuner.get_best_hyperparameters(1)[0].values
    else:
        if args.model in ["gru", "lstm"]:
            inp = keras.Input(shape=(args.window, n_features))
            x = layers.GRU(64)(inp) if args.model == "gru" else layers.LSTM(64)(inp)
            x = layers.Dense(32, activation="relu")(x)
            out = layers.Dense(1)(x)
            model = keras.Model(inp, out)
        else:
            inp = keras.Input(shape=(args.window, n_features))
            x = layers.Flatten()(inp)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dense(64, activation="relu")(x)
            out = layers.Dense(1)(x)
            model = keras.Model(inp, out)

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch,
            callbacks=callbacks
        )
        best_hp = {"manual": True}

    metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = dict(zip(model.metrics_names, metrics))

    model.save(outdir / "model.keras")
    joblib.dump(scaler, outdir / "scaler.joblib")

    config = {
        "value_col": args.value_col,
        "window": args.window,
        "horizon": args.horizon,
        "period": args.period,
        "harmonics": args.harmonics,
        "use_diff": args.use_diff,
        "model": args.model,
        "best_hp": best_hp,
        "metrics": metrics_dict,
        "columns": feat_df.columns.tolist()
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    y_pred = model.predict(X_test).reshape(-1)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.savefig(outdir / "test_pred.png")


if __name__ == "__main__":
    main()
