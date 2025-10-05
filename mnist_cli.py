# mnist_cli.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def build_mnist_model(lr: float = 1e-3) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mnist(
    epochs: int, batch_size: int, lr: float, out_path: Path, plot_curve: bool
) -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")
    x_test = (x_test / 255.0).astype("float32")

    model = build_mnist_model(lr=lr)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    model.save(out_path)
    print(f"✅ Zapisano model: {out_path}")

    if plot_curve:
        if plt is None:
            print("Zainstaluj matplotlib: pip install matplotlib")
            return
        fig_path = Path("learning_curve.png")
        plt.figure()
        plt.plot(history.history.get("accuracy", []), label="train_acc")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("MNIST – learning curve")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"📈 Zapisano wykres: {fig_path}")


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        print(f"❌ Brak pliku modelu: {model_path}")
        sys.exit(2)
    return tf.keras.models.load_model(model_path)


def predict_image(model_path: Path, image_path: Path) -> None:
    if Image is None:
        print("Zainstaluj Pillow: pip install pillow")
        sys.exit(3)

    model = load_model(model_path)

    img = Image.open(image_path).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    probs = model.predict(arr.reshape(1, 28, 28), verbose=0)[0]
    pred = int(np.argmax(probs))
    print(f"🔎 Predykcja: {pred} (p={probs[pred]:.3f})")


def _read_csv_autodetect(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 1:
            df = pd.read_csv(csv_path, sep=";")
        return df
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {csv_path}")
        sys.exit(4)


def train_from_csv(
    csv_path: Path, label_column: str, epochs: int, batch_size: int, lr: float
) -> None:
    df = _read_csv_autodetect(csv_path)

    if label_column not in df.columns:
        print(f"❌ Brak kolumny etykiety '{label_column}'. Kolumny: {list(df.columns)}")
        sys.exit(4)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    y, classes = pd.factorize(df[label_column].astype(str))
    print(f"Klasy: {list(map(str, classes))} → 0..{len(classes)-1}")

    X = pd.get_dummies(df.drop(columns=[label_column]), drop_first=False)
    X = X.fillna(X.median(numeric_only=True)).astype("float32").to_numpy()

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    X_train, y_train, X_test, y_test = X[tr], y[tr], X[te], y[te]

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(np.unique(y)), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Val: acc={acc:.4f}, loss={loss:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser("MNIST / CSV – TensorFlow CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_train = sub.add_parser("train", help="Trening MNIST i zapis modelu .keras")
    sp_train.add_argument("--epochs", type=int, default=5)
    sp_train.add_argument("--batch-size", type=int, default=128)
    sp_train.add_argument("--lr", type=float, default=1e-3)
    sp_train.add_argument("--out", type=Path, default=Path("mnist_model.keras"))
    sp_train.add_argument("--plot", action="store_true")

    sp_pred = sub.add_parser("predict", help="Predykcja cyfry z obrazu")
    sp_pred.add_argument("--model", type=Path, default=Path("mnist_model.keras"))
    sp_pred.add_argument("image", type=Path)

    sp_csv = sub.add_parser("train-csv", help="Trening MLP na danych z CSV")
    sp_csv.add_argument("csv", type=Path)
    sp_csv.add_argument("--label", required=True)
    sp_csv.add_argument("--epochs", type=int, default=20)
    sp_csv.add_argument("--batch-size", type=int, default=256)
    sp_csv.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.cmd == "train":
        train_mnist(args.epochs, args.batch_size, args.lr, args.out, args.plot)
    elif args.cmd == "predict":
        predict_image(args.model, args.image)
    elif args.cmd == "train-csv":
        train_from_csv(args.csv, args.label, args.epochs, args.batch_size, args.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
