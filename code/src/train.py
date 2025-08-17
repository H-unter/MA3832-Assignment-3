import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from model_def import build_model


def find_npz(directory):
    for f in os.listdir(directory):
        if f.endswith(".npz"):
            return os.path.join(directory, f)
    raise FileNotFoundError("No .npz file found in " + directory)


def get_dataset(npz_path, batch_size, shuffle=True, limit=None):
    data = np.load(npz_path, mmap_mode="r")
    assert "image" in data and "label" in data, f"Keys in {npz_path} are {data.files}"
    x = data["image"] if limit is None else data["image"][:limit]
    y = data["label"] if limit is None else data["label"][:limit]
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(x)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1"):
        return True
    if v in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)

    # model architecture choices
    parser.add_argument("--conv1-filters", type=int, default=32)
    parser.add_argument("--conv2-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--pooling", choices=["max", "avg"], default="max")
    parser.add_argument("--use-dropout", type=str2bool, default=True)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=["adam", "adagrad"], default="adam")

    # data / I/O
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument(
        "--model-dir", "--model_dir",
        dest="model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to write model artifacts (accepts --model-dir or --model_dir)."
    )
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.model_dir and str(args.model_dir).startswith("s3://"):
        print(f"WARNING: args.model_dir is an S3 URI ({args.model_dir}). Overriding to local SM_MODEL_DIR: {sm_model_dir}")
        save_model_dir = sm_model_dir
    else:
        save_model_dir = args.model_dir or sm_model_dir
    print("Resolved model save directory:", save_model_dir)
    os.makedirs(save_model_dir, exist_ok=True)

    train_path = find_npz(args.train)
    val_path = find_npz(args.test)

    train_ds = get_dataset(train_path, args.batch_size, shuffle=True, limit=args.limit)
    val_ds = get_dataset(val_path, args.batch_size, shuffle=False, limit=args.limit)

    model = build_model(
        input_shape=(args.height, args.width, args.channels),
        conv1_filters=args.conv1_filters,
        conv2_filters=args.conv2_filters,
        dense_units=args.dense_units,
        use_dropout=args.use_dropout,
        dropout_rate=args.dropout_rate,
        pooling=args.pooling,
    )
    if args.optimizer == "adagrad":
        optimizer_choice = tf.keras.optimizers.Adagrad(learning_rate=args.learning_rate)
    else:
        optimizer_choice = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(
        optimizer=optimizer_choice,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1
    )

    model.save(os.path.join(save_model_dir, "1"))

    final_val = history.history.get("val_binary_accuracy", [None])[-1]
    print("val_accuracy:", round(final_val, 4))

    prec_m = tf.keras.metrics.Precision()
    rec_m  = tf.keras.metrics.Recall()
    auc_m  = tf.keras.metrics.AUC()
    
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_true = tf.cast(yb, tf.int32)
        y_pred = tf.cast(tf.round(probs), tf.int32)
        prec_m.update_state(y_true, y_pred)
        rec_m.update_state(y_true, y_pred)
        auc_m.update_state(y_true, probs)
    prec = float(prec_m.result().numpy())
    rec = float(rec_m.result().numpy())
    auc = float(auc_m.result().numpy())
    f1 = 2 * prec * rec / (prec + rec + 1e-12) # check
    
    print(f"val_precision: {prec:.6f}")
    print(f"val_recall: {rec:.6f}")
    print(f"val_f1: {f1:.6f}")
    print(f"val_auc: {auc:.6f}")


if __name__ == "__main__":
    main()
