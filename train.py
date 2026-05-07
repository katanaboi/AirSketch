"""Reproducible end-to-end training for the AirSketch gesture detection pipeline.

Reads a raw MediaPipe-landmark CSV (42 features + label) and produces a full
set of deployable artifacts into a timestamped run directory:

    <output-dir>/
        normalized.csv
        gesture_classifier.keras
        autoencoder.keras
        label_encoder.pkl
        threshold.json
        model_info.json
        tflite/gesture_classifier.tflite
        tflite/autoencoder.tflite
        eval/confusion_matrix.png
        eval/class_accuracy.png
        eval/confusion_pairs.png
        eval/classification_report.json

Usage:
    python train.py --dataset data/hand_landmarks_dataset.csv
    python train.py --dataset data/hand_landmarks_dataset.csv --output-dir models/runs/custom

Architectures and hyperparameters mirror notebooks 1-3 so the CV / test
accuracy remains comparable.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical


DEFAULT_SEED = 42
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 2


def augment_landmarks(x, y, max_rot_deg=15.0, flip_prob=0.5):
    """Random rotation around wrist (origin post-normalize) + horizontal flip.

    Features arrive already wrist-relative + scale-normalized (see preprocess),
    so the wrist sits at (0, 0) and a plain 2D rotation is geometrically exact.
    """
    batch = tf.shape(x)[0]
    coords = tf.reshape(x, (batch, NUM_LANDMARKS, 2))

    angles = tf.random.uniform((batch,), -max_rot_deg, max_rot_deg) * (np.pi / 180.0)
    cos_a, sin_a = tf.cos(angles), tf.sin(angles)
    rot = tf.stack(
        [tf.stack([cos_a, -sin_a], axis=-1),
         tf.stack([sin_a,  cos_a], axis=-1)],
        axis=-2,
    )
    coords = tf.einsum("bij,bkj->bki", rot, coords)

    flip_mask = tf.cast(tf.random.uniform((batch, 1, 1)) < flip_prob, tf.float32)
    sign = 1.0 - 2.0 * flip_mask
    coords = tf.concat([coords[..., 0:1] * sign, coords[..., 1:2]], axis=-1)

    return tf.reshape(coords, (batch, NUM_FEATURES)), y


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in raw_df.columns if c != "label"]
    if len(feature_cols) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} feature columns, got {len(feature_cols)}"
        )

    coords = raw_df[feature_cols].values.reshape(-1, NUM_LANDMARKS, 2).astype(np.float32)
    coords -= coords[:, 0:1, :]
    distances = np.linalg.norm(coords, axis=2)
    max_d = distances.max(axis=1, keepdims=True)
    max_d = np.where(max_d > 0, max_d, 1.0)
    coords /= max_d[:, :, None]

    out = pd.DataFrame(coords.reshape(-1, NUM_FEATURES), columns=feature_cols)
    out["label"] = raw_df["label"].values
    return out


def create_classifier(input_dim: int, num_classes: int) -> tf.keras.Model:
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(128, kernel_initializer="he_normal", name="hidden_1_dense"),
            tf.keras.layers.BatchNormalization(name="hidden_1_batchnorm"),
            tf.keras.layers.Activation("gelu", name="hidden_1_activation"),
            tf.keras.layers.Dropout(0.3, name="hidden_1_dropout"),
            tf.keras.layers.Dense(64, kernel_initializer="he_normal", name="hidden_2_dense"),
            tf.keras.layers.BatchNormalization(name="hidden_2_batchnorm"),
            tf.keras.layers.Activation("gelu", name="hidden_2_activation"),
            tf.keras.layers.Dropout(0.25, name="hidden_2_dropout"),
            tf.keras.layers.Dense(32, kernel_initializer="he_normal", name="hidden_3_dense"),
            tf.keras.layers.BatchNormalization(name="hidden_3_batchnorm"),
            tf.keras.layers.Activation("gelu", name="hidden_3_activation"),
            tf.keras.layers.Dropout(0.2, name="hidden_3_dropout"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ]
    )


def compile_classifier(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )
    return model


def create_autoencoder(input_dim: int) -> tf.keras.Model:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((input_dim,), name="input_layer"),
            tf.keras.layers.GaussianNoise(0.05, name="input_noise"),
            tf.keras.layers.Dense(64, kernel_initializer="he_normal", name="encoder_dense1"),
            tf.keras.layers.BatchNormalization(name="encoder_bn1"),
            tf.keras.layers.LeakyReLU(negative_slope=0.1, name="encoder_act1"),
            tf.keras.layers.Dense(32, kernel_initializer="he_normal", name="encoder_dense2"),
            tf.keras.layers.BatchNormalization(name="encoder_bn2"),
            tf.keras.layers.LeakyReLU(negative_slope=0.1, name="encoder_act2"),
            tf.keras.layers.Dense(
                8,
                activation=None,
                kernel_initializer="he_normal",
                activity_regularizer=tf.keras.regularizers.L1(1e-5),
                name="bottleneck",
            ),
            tf.keras.layers.Dense(32, kernel_initializer="he_normal", name="decoder_dense1"),
            tf.keras.layers.BatchNormalization(name="decoder_bn1"),
            tf.keras.layers.LeakyReLU(negative_slope=0.1, name="decoder_act1"),
            tf.keras.layers.Dense(64, kernel_initializer="he_normal", name="decoder_dense2"),
            tf.keras.layers.BatchNormalization(name="decoder_bn2"),
            tf.keras.layers.LeakyReLU(negative_slope=0.1, name="decoder_act2"),
            tf.keras.layers.Dense(input_dim, activation="linear", name="output_layer"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
        loss=tf.keras.losses.Huber(),
    )
    return model


def train_classifier_cv(X, y_cat, num_classes, seed, class_weights=None, augment=True):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores, fold_models = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y_cat[train_idx], y_cat[val_idx]
        print(f"  fold {fold + 1}/5 ...", end=" ", flush=True)
        model = compile_classifier(create_classifier(X_tr.shape[1], num_classes))

        train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        train_ds = train_ds.shuffle(buffer_size=len(X_tr), seed=seed).batch(32)
        if augment:
            train_ds = train_ds.map(augment_landmarks, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        history = model.fit(
            train_ds,
            validation_data=(X_va, y_va),
            epochs=50, verbose=0,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=10, restore_best_weights=True
                )
            ],
        )
        val_acc = float(max(history.history["val_accuracy"]))
        cv_scores.append(val_acc)
        fold_models.append(model)
        print(f"val_acc={val_acc:.4f}")
    best = int(np.argmax(cv_scores))
    print(f"  best fold: {best + 1} (val_acc={cv_scores[best]:.4f})")
    return fold_models[best], cv_scores, best


def train_autoencoder_cv(X, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores, fold_models, fold_train_splits = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[train_idx], X[val_idx]
        print(f"  fold {fold + 1}/5 ...", end=" ", flush=True)
        model = create_autoencoder(X_tr.shape[1])
        history = model.fit(
            X_tr, X_tr, validation_data=(X_va, X_va),
            epochs=20, batch_size=32, verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            ],
        )
        val_loss = float(min(history.history["val_loss"]))
        cv_scores.append(val_loss)
        fold_models.append(model)
        fold_train_splits.append(X_tr)
        print(f"val_loss={val_loss:.6f}")
    best = int(np.argmin(cv_scores))
    print(f"  best fold: {best + 1} (val_loss={cv_scores[best]:.6f})")
    return fold_models[best], fold_train_splits[best], cv_scores, best


def compute_threshold(autoencoder, X_train, percentile=95) -> float:
    recon = autoencoder.predict(X_train, verbose=0)
    errors = np.mean(np.square(X_train - recon), axis=1)
    return float(np.percentile(errors, percentile))


def convert_to_tflite(model, out_path: Path, optimize: bool) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tfl)


def evaluate_and_plot(model, X_test, y_test_cat, class_names, eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    class_accs = []
    for i in range(len(class_names)):
        mask = y_true == i
        class_accs.append(float(np.mean(y_pred[mask] == y_true[mask])) if mask.any() else 0.0)
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accs)
    plt.title("Class-wise Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(eval_dir / "class_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.fill_diagonal(cmn, 0)
    pairs = [
        (i, j, cmn[i, j])
        for i in range(cmn.shape[0])
        for j in range(cmn.shape[1])
        if cmn[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    top = pairs[:5]
    plt.figure(figsize=(10, 6))
    if top:
        labels = [f"{class_names[p[0]]} \u2192 {class_names[p[1]]}" for p in top]
        plt.barh(labels, [p[2] for p in top])
        plt.title("Top Confused Pairs")
        plt.xlabel("Rate")
    else:
        plt.text(0.5, 0.5, "No confusion", ha="center", va="center")
        plt.title("Top Confused Pairs")
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_pairs.png", dpi=200, bbox_inches="tight")
    plt.close()

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    (eval_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    return float(np.mean(y_pred == y_true)), report


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end trainer for the AirSketch detection pipeline."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Raw landmarks CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: models/runs/<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable rotation + flip augmentation on the classifier training set.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or (Path("models/runs") / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")

    print("\n[1/5] Preprocessing (wrist-relative + scale normalization)...")
    raw = pd.read_csv(args.dataset)
    normalized = preprocess(raw)
    normalized_path = out_dir / "normalized.csv"
    normalized.to_csv(normalized_path, index=False)
    print(f"  wrote {normalized_path.name} ({len(normalized)} rows)")

    feature_cols = [c for c in normalized.columns if c != "label"]
    X = normalized[feature_cols].values.astype(np.float32)
    y = normalized["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes=num_classes)

    X_trainval, X_test, y_trainval_cat, y_test_cat = train_test_split(
        X, y_cat, test_size=0.2, random_state=args.seed, stratify=y_enc
    )
    print(f"  classes ({num_classes}): {list(le.classes_)}")
    print(f"  train/val: {X_trainval.shape[0]}, test: {X_test.shape[0]}")

    y_trainval_enc = np.argmax(y_trainval_cat, axis=1)
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_trainval_enc
    )
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    augment = not args.no_augment
    print(
        f"\n[2/5] Training classifier (5-fold CV, up to 50 epochs/fold, "
        f"augment={augment}, class_weight=balanced)..."
    )
    best_clf, clf_cv, clf_best = train_classifier_cv(
        X_trainval, y_trainval_cat, num_classes, args.seed,
        class_weights=class_weights, augment=augment,
    )

    print("\n[3/5] Training autoencoder (5-fold CV, up to 20 epochs/fold)...")
    best_ae, ae_best_train, ae_cv, ae_best = train_autoencoder_cv(X_trainval, args.seed)

    print("\n[4/5] Computing reconstruction threshold...")
    threshold = compute_threshold(best_ae, ae_best_train, percentile=95)
    print(f"  threshold (95th pct): {threshold:.6f}")

    print("\n[5/5] Exporting artifacts + evaluating on held-out test set...")
    best_clf.save(out_dir / "gesture_classifier.keras")
    best_ae.save(out_dir / "autoencoder.keras")
    joblib.dump(le, out_dir / "label_encoder.pkl")
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))

    convert_to_tflite(best_clf, out_dir / "tflite" / "gesture_classifier.tflite", optimize=True)
    convert_to_tflite(best_ae, out_dir / "tflite" / "autoencoder.tflite", optimize=False)

    test_acc, _ = evaluate_and_plot(
        best_clf, X_test, y_test_cat, list(le.classes_), out_dir / "eval"
    )
    print(f"  test accuracy: {test_acc:.4f}")

    info = {
        "timestamp": timestamp,
        "git_sha": get_git_sha(),
        "dataset": str(args.dataset.resolve()),
        "dataset_rows": int(len(normalized)),
        "seed": args.seed,
        "classes": list(le.classes_),
        "num_classes": num_classes,
        "classifier": {
            "cv_val_accuracy": [float(s) for s in clf_cv],
            "mean_cv_accuracy": float(np.mean(clf_cv)),
            "best_fold": clf_best + 1,
            "test_accuracy": float(test_acc),
            "augment": augment,
            "class_weight": "balanced",
        },
        "autoencoder": {
            "cv_val_loss": [float(s) for s in ae_cv],
            "mean_cv_val_loss": float(np.mean(ae_cv)),
            "best_fold": ae_best + 1,
            "threshold": threshold,
        },
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))

    print("\n" + "=" * 64)
    print(f"Done. Artifacts in: {out_dir.resolve()}")
    print(f"  mean CV accuracy : {np.mean(clf_cv):.4f}")
    print(f"  test accuracy    : {test_acc:.4f}")
    print(f"  AE mean CV loss  : {np.mean(ae_cv):.6f}")
    print(f"  threshold        : {threshold:.6f}")
    print("=" * 64)
    print("\nTo deploy this run to the live app, copy these four files:")
    rel = out_dir.as_posix()
    print(f"  {rel}/tflite/gesture_classifier.tflite  ->  models/tflite/gesture_classifier.tflite")
    print(f"  {rel}/tflite/autoencoder.tflite         ->  models/tflite/autoencoder.tflite")
    print(f"  {rel}/label_encoder.pkl                 ->  models/label_encoder.pkl")
    print(f"  {rel}/threshold.json                    ->  models/threshold.json")


if __name__ == "__main__":
    main()
