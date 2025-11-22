import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras_tuner as kt

RNG = 42  # stałe ziarno losowe, żeby wyniki były powtarzalne

# nazwy kolumn, w takiej kolejności jak w pliku wine.data
COLS = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
]

# parser do obsługi dwóch komend: train i predict
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)

# trenowanie + tuning
p_train = sub.add_parser("train")
p_train.add_argument("--csv", required=True)          # ścieżka do wine.data
p_train.add_argument("--outdir", default="artifacts") # gdzie zapisywać wyniki
p_train.add_argument("--epochs", type=int, default=30)
p_train.add_argument("--batch", type=int, default=16)
p_train.add_argument("--max_trials", type=int, default=10)

# pojedyncza predykcja
p_pred = sub.add_parser("predict")
p_pred.add_argument("--model_dir", default="artifacts/models")
for f in COLS[1:]:
    p_pred.add_argument(f"--{f}", type=float, required=True)

args = parser.parse_args()


def load_data(csv_path):
    """Wczytanie wine.data do DataFrame i przetasowanie wierszy."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = COLS
    df = df.sample(frac=1.0, random_state=RNG).reset_index(drop=True)
    return df


def prepare_data(df):
    """Podział na X/y, one-hot i train/val (80/20)."""
    X = df.drop(columns=["class"]).to_numpy(np.float32)
    y = df["class"].to_numpy(np.int32)
    y_oh = tf.keras.utils.to_categorical(y - 1, num_classes=3)

    X_tr, X_val, y_tr, y_val, ytr_oh, yval_oh = train_test_split(
        X, y, y_oh, test_size=0.2, stratify=y, random_state=RNG
    )
    return X_tr, X_val, y_tr, y_val, ytr_oh, yval_oh


def make_model(input_shape, normalizer,
               units1=32, units2=32,
               learning_rate=0.01,
               activation="relu"):
    """Buduje i kompiluje model dla zadanych parametrów."""
    inputs = tf.keras.Input(shape=input_shape)
    x = normalizer(inputs)  # normalizacja jako pierwsza warstwa

    x = tf.keras.layers.Dense(
        units1,
        activation=activation,
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.Dense(
        units2,
        activation=activation,
        kernel_initializer="he_normal"
    )(x)

    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9
    )
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_model_hp(hp, input_shape, normalizer):
    """
    Wersja make_model używana przez Keras Tuner.
    hp dostarcza wartości hiperparametrów.
    """
    units1 = hp.Int("units1", 16, 128, step=16)
    units2 = hp.Int("units2", 16, 128, step=16)
    lr = hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    activation = hp.Choice("activation", ["relu", "tanh"])
    return make_model(input_shape, normalizer, units1, units2, lr, activation)


if args.cmd == "train":
    # 1) dane
    df = load_data(args.csv)
    X_tr, X_val, y_tr, y_val, ytr_oh, yval_oh = prepare_data(df)
    input_shape = (X_tr.shape[1],)  # u nas (13,)

    # 2) normalizacja w Kerasie
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(X_tr)

    # 3) katalogi na wyniki
    os.makedirs(args.outdir + "/plots", exist_ok=True)
    os.makedirs(args.outdir + "/models", exist_ok=True)

    # 4) model bazowy (baseline) – stałe parametry
    baseline_model = make_model(
        input_shape, normalizer,
        units1=32,
        units2=32,
        learning_rate=0.01,
        activation="relu"
    )

    hist = baseline_model.fit(
        X_tr, ytr_oh,
        validation_data=(X_val, yval_oh),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=0
    )

    # wykresy dla baseline
    epochs_list = range(1, len(hist.history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs_list, hist.history["loss"], label="loss")
    plt.plot(epochs_list, hist.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(args.outdir + "/plots/baseline_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs_list, hist.history["accuracy"], label="train_acc")
    plt.plot(epochs_list, hist.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.savefig(args.outdir + "/plots/baseline_acc.png")
    plt.close()

    _, baseline_acc = baseline_model.evaluate(X_val, yval_oh, verbose=0)
    baseline_model.save(args.outdir + "/models/baseline_model.keras")

    # 5) Keras Tuner – RandomSearch
    def tuner_builder(hp):
        return build_model_hp(hp, input_shape, normalizer)

    tuner = kt.RandomSearch(
        tuner_builder,
        objective="val_accuracy",
        max_trials=args.max_trials,
        directory=args.outdir,
        project_name="tuner_wine"
    )

    tuner.search(
        X_tr, ytr_oh,
        validation_data=(X_val, yval_oh),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=0
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    _, tuned_acc = best_model.evaluate(X_val, yval_oh, verbose=0)

    # 6) macierz pomyłek dla najlepszego modelu
    preds = best_model.predict(X_val, verbose=0)
    preds = np.argmax(preds, axis=1)
    true = y_val - 1
    cm = confusion_matrix(true, preds)

    # zapis najlepszego modelu
    best_model.save(args.outdir + "/models/best_model.keras")

    print("Najlepszy model:")
    best_model.summary()

    # 7) zapis metryk do raportu
    summary = {
        "baseline_val_acc": float(baseline_acc),
        "tuned_val_acc": float(tuned_acc),
        "best_hyperparameters": best_hp.values,
        "confusion_matrix": cm.tolist()
    }

    with open(args.outdir + "/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Baseline acc:", baseline_acc)
    print("Tuned acc:", tuned_acc)
    print("Najlepsze hiperparametry:", best_hp.values)


elif args.cmd == "predict":
    # predykcja jednego przykładu z linii komend
    model = tf.keras.models.load_model(args.model_dir + "/best_model.keras")
    x = np.array([[getattr(args, c) for c in COLS[1:]]], dtype=np.float32)
    p = model.predict(x, verbose=0)[0]
    print(int(np.argmax(p)) + 1)
