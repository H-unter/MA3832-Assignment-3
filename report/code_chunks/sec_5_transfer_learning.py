# transfer_learning.ipynb

metric_definitions = [
    {"Name": "val_auc",       "Regex": "val_auc: ([0-9\\.]+)"},
    {"Name": "val_f1",        "Regex": "val_f1: ([0-9\\.]+)"},
    {"Name": "val_precision", "Regex": "val_precision: ([0-9\\.]+)"},
    {"Name": "val_recall",    "Regex": "val_recall: ([0-9\\.]+)"},
    {"Name": "val_accuracy",  "Regex": "val_accuracy: ([0-9\\.]+)"},
]

estimator = tf(
    entry_point="train_transfer.py",
    source_dir="src",
    role=role,
    instance_type="ml.c5.2xlarge",
    instance_count=1,
    framework_version="2.14",
    py_version="py310",
    output_path=f"s3://{bucket}/model_output",
    hyperparameters={
        "epochs": 3,
        "height": 224, "width": 224, "channels": 3,
        "batch-size": 1,                 
        "learning-rate": 5e-5,          
        "dropout-rate": 0.2,
        "unfreeze-fraction": 0.10,
    },
    metric_definitions=metric_definitions,
)

job_name = f"transfer-learning-{time.strftime('%Y%m%d-%H%M%S')}"
estimator.fit({"train": train_input, "test": test_input}, job_name=job_name)

# src/transfer_model.py

def build_transfer_model(input_shape=(224, 224, 3), dropout_rate=0.2, unfreeze_fraction=0.0):
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base.trainable = False
    if unfreeze_fraction > 0:
        n = len(base.layers)
        k = max(1, int(round(n * float(unfreeze_fraction))))
        for layer in base.layers[-k:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = inputs                         # keep values in [0,255] float
    x = base(x)                        # EfficientNet applies its own Rescaling inside
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)