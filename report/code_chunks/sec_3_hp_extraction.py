sm = boto3.client("sagemaker")
training_job_name = "ph-17-250815-1154-018-22526da0"  
tj = sm.describe_training_job(TrainingJobName=training_job_name)
raw_hps = dict(tj["HyperParameters"])  # strings
raw_hps

# further cleaning the result is also required

train_input = TrainingInput(train_npz, input_mode="File", content_type="application/x-npz")
test_input  = TrainingInput(test_npz,  input_mode="File", content_type="application/x-npz")

estimator = tf(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.c5.2xlarge",
    instance_count=1,
    framework_version="2.14",
    py_version="py310",
    output_path=f"s3://{bucket}/aiornot/model_output",
    # keep the static shape/run params you use + inject the tuned ones
    hyperparameters={
        "epochs": 5, "height": 512, "width": 512, "channels": 3,
        **typed_hps,  # tuned values win if keys overlap
    },
    metric_definitions=[
        {"Name":"val_auc","Regex":r"val_auc: ([0-9\.]+)"},
        {"Name":"val_f1","Regex":r"val_f1: ([0-9\.]+)"},
        {"Name":"val_precision","Regex":r"val_precision: ([0-9\.]+)"},
        {"Name":"val_recall","Regex":r"val_recall: ([0-9\.]+)"},
        {"Name":"val_accuracy","Regex":r"val_accuracy: ([0-9\.]+)"},
    ],
)
# Creating training-job with name like bestparams-refit-20250815-091227
job_name = "bestparams-refit-" + time.strftime("%Y%m%d-%H%M%S")
estimator.fit(
    {"train": train_input, "test": test_input},
    job_name=job_name
)