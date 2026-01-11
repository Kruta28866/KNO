#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from tensorflow import keras


def add_fourier_features(df, t, period, harmonics):
    for k in range(1, harmonics + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    art = Path(args.artifacts)
    config = json.loads((art / "config.json").read_text())

    model = keras.models.load_model(art / "model.keras")
    scaler = joblib.load(art / "scaler.joblib")

    df = pd.read_csv(args.history)
    value_col = config["value_col"]

    series = df[[value_col]].copy()

    if config["use_diff"]:
        series["value"] = series[value_col].diff()
        series = series.drop(columns=[value_col]).dropna().reset_index(drop=True)
        target_col = "value"
        last_level = float(df[value_col].iloc[-1])
    else:
        series = series.rename(columns={value_col: "value"})
        target_col = "value"
        last_level = None

    window = config["window"]
    period = config["period"]
    harmonics = config["harmonics"]
    cols = config["columns"]

    feat_df = add_fourier_features(series.copy(), np.arange(len(series)), period, harmonics)
    feat_df = feat_df[cols]

    feat_scaled = scaler.transform(feat_df)
    feat_scaled = np.array(feat_scaled)

    preds = []

    for step in range(args.n):
        x = feat_scaled[-window:][None, :, :]
        y_pred = float(model.predict(x, verbose=0)[0, 0])
        preds.append(y_pred)

        next_t = len(feat_df) + step
        new_row = {c: 0.0 for c in cols}
        new_row[target_col] = y_pred

        for k in range(1, harmonics + 1):
            new_row[f"sin_{k}"] = np.sin(2 * np.pi * k * next_t / period)
            new_row[f"cos_{k}"] = np.cos(2 * np.pi * k * next_t / period)

        new_row_df = pd.DataFrame([new_row])[cols]
        new_scaled = scaler.transform(new_row_df)[0]
        feat_scaled = np.vstack([feat_scaled, new_scaled])

    if config["use_diff"]:
        level = []
        cur = last_level
        for d in preds:
            cur += d
            level.append(cur)
        out = pd.DataFrame({"pred_diff": preds, "pred_level": level})
    else:
        out = pd.DataFrame({"pred": preds})

    out.to_csv(args.result, index=False)


if __name__ == "__main__":
    main()
