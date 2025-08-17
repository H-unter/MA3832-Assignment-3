main_model_s3_path = "s3://sagemaker-ap-southeast-2-838084669510/aiornot/model_output/bestparams-refit-20250816-094717/output/model.tar.gz"
main_model = TensorFlowModel(
    model_data=main_model_s3_path,
    role=role,
    framework_version="2.14"
)
try: # has not yet been deployed
    main_model_predictor = main_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="final-model-endpoint"
    )
except Exception: # has already been deployed, so just call the existing endpoint
    main_model_predictor = TensorFlowPredictor(
        endpoint_name="final-model-endpoint",
        sagemaker_session=sess
    )

# similarly, the transfer model endpoint is also invoked

test_path = f"s3://{bucket}/{prefix}/holdout_test/holdout_test.npz"

with fs.open(test_path, "rb") as f:
    d = np.load(f)
    X = d["image"].astype("float32") 
    y_true = np.asarray(d["label"], dtype=int).ravel()
    print("data loaded")

def predict_batches(pred, X, bs=4):
    probs = []
    for i in range(0, len(X), bs):
        out = pred.predict(X[i:i+bs].tolist())
        p = np.array(out.get("predictions", out)).reshape(-1)  # shape (bs,)
        probs.append(p)
        print(f"{i}/{len(X)}")
    probs = np.concatenate(probs)
    predictions = (probs >= 0.5).astype(int)
    return predictions


y_test_main_model = predict_batches(main_model_predictor, X)
y_test_transfer_model = predict_batches(transfer_model_predictor, X)