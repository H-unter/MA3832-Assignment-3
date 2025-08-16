import argparse, os, math, numpy as np
import tensorflow as tf
from transfer_model import build_transfer_model

def find_npz(directory):
    for f in os.listdir(directory):
        if f.endswith(".npz"):
            return os.path.join(directory, f)
    raise FileNotFoundError(f"No .npz file found in {directory}")

def npz_gen(npz_path):
    d = np.load(npz_path, mmap_mode="r")
    X, y = d["image"], d["label"]
    n = X.shape[0]
    def gen():
        for i in range(n):
            yield X[i], y[i]
    return gen, n

def make_ds(npz_path, batch, height, width, shuffle=False, buffer=128):
    gen, n = npz_gen(npz_path)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if shuffle:
        ds = ds.shuffle(min(buffer, n))
    ds = ds.map(
        lambda xi, yi: (tf.image.resize(tf.cast(xi, tf.float32), [height, width]), yi),
        num_parallel_calls=1,
    ).batch(batch).repeat().prefetch(1)
    return ds, n

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dropout-rate", type=float, default=0.2)
    p.add_argument("--unfreeze-fraction", type=float, default=0.10)  # ← 10%
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--test",  type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", "--model_dir", dest="model_dir",
                   type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = p.parse_args()

    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    save_model_dir = sm_model_dir if str(args.model_dir).startswith("s3://") else (args.model_dir or sm_model_dir)
    if str(args.model_dir).startswith("s3://"):
        print(f"WARNING: args.model_dir is an S3 URI ({args.model_dir}). Overriding to local SM_MODEL_DIR: {sm_model_dir}")
    os.makedirs(save_model_dir, exist_ok=True)
    print("Resolved model save directory:", save_model_dir)

    train_path = find_npz(args.train)
    test_path  = find_npz(args.test)
    train_ds, n_train = make_ds(train_path, args.batch_size, args.height, args.width, shuffle=True)
    val_ds,   n_val   = make_ds(test_path,  args.batch_size, args.height, args.width, shuffle=False)

    steps_per_epoch = max(1, math.ceil(n_train / args.batch_size))
    val_steps       = max(1, math.ceil(n_val   / args.batch_size))
    print("train_samples:", n_train, "val_samples:", n_val, "steps:", steps_per_epoch, "val_steps:", val_steps)

    model = build_transfer_model(
        input_shape=(args.height, args.width, args.channels),
        dropout_rate=args.dropout_rate,
        unfreeze_fraction=args.unfreeze_fraction
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=1,
        workers=1, use_multiprocessing=False,
    )

    model.save(os.path.join(save_model_dir, "1"))

    geth = history.history.get
    acc = float(geth("val_accuracy",  [0.0])[-1])
    pre = float(geth("val_precision", [0.0])[-1])
    rec = float(geth("val_recall",    [0.0])[-1])
    auc = float(geth("val_auc",       [0.0])[-1])
    f1  = (2 * pre * rec) / (pre + rec + 1e-12)

    print(f"val_precision: {pre:.6f}")
    print(f"val_recall: {rec:.6f}")
    print(f"val_f1: {f1:.6f}")
    print(f"val_auc: {auc:.6f}")
    print(f"val_accuracy: {acc:.6f}")

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    main()
