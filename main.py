import argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = 42
COLS = ["class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
        "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins",
        "color_intensity","hue","od280_od315_of_diluted_wines","proline"]

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)

p_train = sub.add_parser("train")
p_train.add_argument("--csv", required=True)             # np. Data/wine/wine.data
p_train.add_argument("--outdir", default="artifacts")    # gdzie zapisywać
p_train.add_argument("--epochs", type=int, default=200)
p_train.add_argument("--batch", type=int, default=16)

p_pred = sub.add_parser("predict")
p_pred.add_argument("--model_dir", default="artifacts/models")
for f in COLS[1:]:
    p_pred.add_argument(f"--{f}", type=float, required=True)

args = parser.parse_args()

if args.cmd == "train":
    # 1) wczytanie + nadanie nagłówków + tasowanie
    df = pd.read_csv(args.csv, header=None); df.columns = COLS
    df = df.sample(frac=1.0, random_state=RNG).reset_index(drop=True)

    # 2) cechy/etykiety
    X = df.drop(columns=["class"]).to_numpy(np.float32)
    y = df["class"].to_numpy(np.int32)                   # 1..3
    y_oh = tf.keras.utils.to_categorical(y-1, num_classes=3)  # one-hot

    # 3) podział + standaryzacja (fit na train)
    X_tr, X_te, y_tr, y_te, ytr_oh, yte_oh = train_test_split(
        X, y, y_oh, test_size=0.2, stratify=y, random_state=RNG
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr); X_te = scaler.transform(X_te)

    # 4) dwa modele
    modelA = tf.keras.Sequential([
        tf.keras.layers.Input((13,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(3, activation="softmax")
    ], name="A")
    modelA.compile(optimizer=tf.keras.optimizers.SGD(0.01, momentum=0.9),
                   loss="categorical_crossentropy", metrics=["accuracy"])

    modelB = tf.keras.Sequential([
        tf.keras.layers.Input((13,)),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(3, activation="softmax")
    ], name="B")
    modelB.compile(optimizer=tf.keras.optimizers.SGD(0.005, momentum=0.9),
                   loss="categorical_crossentropy", metrics=["accuracy"])

    # 5) uczenie + wykresy
    hA = modelA.fit(X_tr, ytr_oh, validation_data=(X_te, yte_oh),
                    epochs=args.epochs, batch_size=args.batch, verbose=0)
    hB = modelB.fit(X_tr, ytr_oh, validation_data=(X_te, yte_oh),
                    epochs=args.epochs, batch_size=max(8, args.batch//2), verbose=0)

    import os, json as _json
    os.makedirs(f"{args.outdir}/plots", exist_ok=True)
    os.makedirs(f"{args.outdir}/models", exist_ok=True)

    for name, h in [("modelA", hA), ("modelB", hB)]:
        e = range(1, len(h.history["loss"])+1)
        plt.figure(); plt.plot(e, h.history["loss"], label="loss"); plt.plot(e, h.history["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{args.outdir}/plots/{name}_loss.png"); plt.close()
        plt.figure(); plt.plot(e, h.history["accuracy"], label="train_acc"); plt.plot(e, h.history["val_accuracy"], label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{args.outdir}/plots/{name}_acc.png"); plt.close()

    # 6) test accuracy + wybór najlepszego + zapis
    _, accA = modelA.evaluate(X_te, yte_oh, verbose=0)
    _, accB = modelB.evaluate(X_te, yte_oh, verbose=0)
    best, best_model = ("A", modelA) if accA >= accB else ("B", modelB)

    best_model.save(f"{args.outdir}/models/best_model.keras")
    with open(f"{args.outdir}/models/scaler.json","w") as f:
        _json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)
    with open(f"{args.outdir}/training_summary.json","w") as f:
        _json.dump({"epochs": args.epochs, "batch": args.batch,
                    "A_test_acc": float(accA), "B_test_acc": float(accB),
                    "best": best}, f, indent=2)
    print(f"OK. Test acc: A={accA:.4f}, B={accB:.4f}. Best={best}. Artefakty w {args.outdir}/")

elif args.cmd == "predict":
    # 7) predykcja: wczytaj model i scaler, przyjmij 13 cech, zwróć klasę 1..3
    model = tf.keras.models.load_model(f"{args.model_dir}/best_model.keras")
    with open(f"{args.model_dir}/scaler.json") as f:
        sc = json.load(f)
    x = np.array([[getattr(args, c) for c in COLS[1:]]], dtype=np.float32)
    x = (x - np.array(sc["mean"], np.float32)) / np.array(sc["scale"], np.float32)
    p = model.predict(x, verbose=0)[0]
    print(int(np.argmax(p)) + 1)
