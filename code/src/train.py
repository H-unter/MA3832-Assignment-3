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

def get_dataset(npz_path, batch_size, shuffle=True, limit=256):
    data = np.load(npz_path, mmap_mode="r")
    assert "image" in data and "label" in data, f"Keys in {npz_path} are {data.files}"
    x = data["image"][:limit]
    y = data["label"][:limit]
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, limit))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("Loaded", npz_path, "with shape:", x.shape, y.shape, "dtype:", y.dtype)

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dense", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    
    args = parser.parse_args()

    train_path = find_npz(args.train)
    val_path = find_npz(args.test)
    
    print("Looking for training file at:", train_path)
    print("Looking for validation file at:", val_path)
    print("Training dir contents:", os.listdir(args.train))
    print("Validation dir contents:", os.listdir(args.test))
    
    train_ds = get_dataset(train_path, args.batch_size, shuffle=True, limit=256)
    val_ds = get_dataset(val_path,   args.batch_size, shuffle=False, limit=256)

    model = build_model(
        input_shape=(args.height, args.width, args.channels),
        num_classes=args.num_classes,
        dense=args.dense,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2,
    )

    # Save final model
    model.save(os.path.join(args.model_dir, "1"))

    # Emit final metric value for SageMaker HPO tracking
    final_val_acc = history.history["val_binary_accuracy"][-1]
    print("val_accuracy:", round(final_val_acc, 4))


if __name__ == "__main__":
    main()
