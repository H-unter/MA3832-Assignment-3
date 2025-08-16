# main.ipynb

estimator = tf(
    entry_point="train.py",
    source_dir="src",  
    role=role,
    use_spot_instances=True,  # save money
    instance_type="ml.c5.2xlarge",
    instance_count=1,
    framework_version="2.14",
    py_version="py310",
    hyperparameters={
        "epochs": 3,
        "height": 512,
        "width": 512,
        "channels": 3
    },
    output_path=s3_output_location
)

hyperparameter_ranges = {
    "learning-rate": ContinuousParameter(1e-4, 1e-2, scaling_type="Logarithmic"),
    "dropout-rate": ContinuousParameter(0.0, 0.5),                 
    "batch-size": IntegerParameter(4, 8),                         
    "conv1-filters": IntegerParameter(16, 128),
    "conv2-filters": IntegerParameter(32, 256),
    "dense-units": IntegerParameter(64, 512),
    "pooling": CategoricalParameter(["max", "avg"]),
    "use-dropout": CategoricalParameter(["true", "false"]),
    "optimizer": CategoricalParameter(["adam", "adagrad"]),
}

metric_definitions = [
    {"Name": "val_auc",       "Regex": "val_auc: ([0-9\\.]+)"},
    {"Name": "val_f1",        "Regex": "val_f1: ([0-9\\.]+)"},
    {"Name": "val_precision", "Regex": "val_precision: ([0-9\\.]+)"},
    {"Name": "val_recall",    "Regex": "val_recall: ([0-9\\.]+)"},
    {"Name": "val_accuracy",  "Regex": "val_accuracy: ([0-9\\.]+)"},
]

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="val_f1",
    strategy='Hyperband',
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_parallel_jobs=5,
    objective_type="Maximize",
    # early_stopping_type="Auto", # not supported for hyperband strategy, since it gets rid of unpromising trials itself
    max_jobs=20,
    base_tuning_job_name="ph-17",
)

tuner.fit({
    "train": small_train_input,
    "test": test_input,
})


# src/train.py
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