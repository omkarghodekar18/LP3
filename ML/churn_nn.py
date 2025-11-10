
"""
Bank Customer Churn - Simple Neural Network Classifier
------------------------------------------------------
Assumes Kaggle's "Churn_Modelling.csv" schema:
- Target column: Exited (0/1)
- Drop identifiers: RowNumber, CustomerId, Surname
- Features: numeric + one-hot for Geography; Gender mapped to 0/1
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

RANDOM_STATE = 42

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    required_cols = {
        "RowNumber","CustomerId","Surname","CreditScore","Geography","Gender","Age",
        "Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    return df

def preprocess(df: pd.DataFrame):
    # Drop identifiers
    df = df.drop(columns=["RowNumber","CustomerId","Surname"])

    # Gender to 0/1
    df["Gender"] = df["Gender"].map({"Female":0, "Male":1}).astype("int64")

    # One-hot encode Geography (drop first to avoid perfect multicollinearity)
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    # Ensure target is int
    y = df["Exited"].astype("int64").values
    X = df.drop(columns=["Exited"])

    # Train/Val/Test split (70/15/15 stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Scale numeric features only (keep columns order for reuse)
    numeric_cols = [
        "CreditScore","Gender","Age","Tenure","Balance","NumOfProducts",
        "HasCrCard","IsActiveMember","EstimatedSalary"
    ]
    # Include geography one-hot cols too (they are already 0/1, scaling not harmful)
    geo_cols = [c for c in X.columns if c.startswith("Geography_")]
    all_cols = numeric_cols + geo_cols

    scaler = StandardScaler()
    scaler.fit(X_train[all_cols])

    X_train_s = X_train.copy()
    X_val_s   = X_val.copy()
    X_test_s  = X_test.copy()

    X_train_s[all_cols] = scaler.transform(X_train[all_cols])
    X_val_s[all_cols]   = scaler.transform(X_val[all_cols])
    X_test_s[all_cols]  = scaler.transform(X_test[all_cols])

    meta = {
        "feature_order": list(X.columns),
        "scaled_cols": all_cols
    }

    return (X_train_s.values, y_train,
            X_val_s.values, y_val,
            X_test_s.values, y_test,
            meta, scaler)

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model

def plot_and_save_roc(y_true, y_prob, out_png: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Bank Churn")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Churn_Modelling.csv", help="Path to Kaggle CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--outdir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load
    df = load_data(args.csv)

    # 2) Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, meta, scaler = preprocess(df)

    # 3) Build
    model = build_model(input_dim=X_train.shape[1])

    # Class weights (helpful for class imbalance)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # Callbacks
    ckpt_path = os.path.join(args.outdir, "best_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    # 4) Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        verbose=1,
        callbacks=callbacks
    )

    # 5) Evaluate
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Reports
    report = classification_report(y_test, y_pred, digits=3)
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save ROC
    roc_path = os.path.join(args.outdir, "roc_curve.png")
    plot_and_save_roc(y_test, y_prob, roc_path)

    # Save scaler + meta
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save final model
    final_model_path = os.path.join(args.outdir, "final_model.keras")
    model.save(final_model_path)

    # Print concise summary
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test ROC AUC : {auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:\n", report)
    print(f"\nArtifacts saved in: {args.outdir}")
    print(f"- Best model checkpoint: {ckpt_path}")
    print(f"- Final model: {final_model_path}")
    print(f"- Meta (features): {os.path.join(args.outdir,'meta.json')}")
    print(f"- ROC curve: {roc_path}")
    print(f"- Classification report: {os.path.join(args.outdir,'classification_report.txt')}")

if __name__ == "__main__":
    # Make TensorFlow quieter
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
