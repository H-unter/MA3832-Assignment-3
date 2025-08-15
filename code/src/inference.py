import io
import json
import os
import boto3
import numpy as np

# Helper: parse s3://bucket/key
def _parse_s3_uri(s3_uri):
    assert s3_uri.startswith("s3://")
    _, _, path = s3_uri.partition("s3://")
    bucket, _, key = path.partition("/")
    return bucket, key

# Called by the SageMaker python service to convert incoming request -> payload for TF Serving
def input_handler(data, content_type):
    """
    data: bytes (raw request body)
    content_type: e.g. 'application/json'
    We return a tuple (body_bytes, content_type) that will be forwarded to TF Serving (REST).
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json.")

    # decode bytes -> dict
    if isinstance(data, (bytes, bytearray)):
        payload = json.loads(data.decode("utf-8"))
    else:
        payload = json.loads(data)

    # If caller gave an s3_path, download the file (don't call .decode() on StreamingBody)
    if isinstance(payload, dict) and "s3_path" in payload:
        s3_uri = payload["s3_path"]
        bucket, key = _parse_s3_uri(s3_uri)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        body_bytes = obj["Body"].read()                # <-- read(), not .decode()
        # np.load from bytes
        with io.BytesIO(body_bytes) as b:
            npz = np.load(b, allow_pickle=False)
            # expected keys: 'image' and 'label'
            images = npz["image"]                      # shape (N,H,W,C)
            # Cast to float and tolist for JSON
            instances = images.astype(float).tolist()
            out = {"instances": instances}
            out_bytes = json.dumps(out).encode("utf-8")
            return out_bytes, "application/json"

    # If caller passed instances directly:
    if isinstance(payload, dict) and "instances" in payload:
        out_bytes = json.dumps({"instances": payload["instances"]}).encode("utf-8")
        return out_bytes, "application/json"

    # If payload is already a list of instances
    if isinstance(payload, list):
        out_bytes = json.dumps({"instances": payload}).encode("utf-8")
        return out_bytes, "application/json"

    raise ValueError("JSON must contain 's3_path' or 'instances' or be a list of instances.")


# Called by the TF serving proxy to convert TF Serving response -> client response
def output_handler(response, accept):
    """
    response: requests.Response from TF Serving (the internal proxy)
    accept: accept header (e.g. application/json)
    Return: (payload_bytes, content_type)
    """
    # response.content is bytes
    content_type = "application/json"
    try:
        raw = response.content
        # many TF Serving images send JSON like {"predictions": [[0.12], [0.99], ...]}
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded)
        # Normalize shape: return flat list of probabilities if nested
        preds = parsed.get("predictions", parsed)
        # If predictions are nested lists of length-1, flatten
        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list) and len(preds[0]) == 1:
            preds = [float(p[0]) for p in preds]
        else:
            # try to flatten numeric arrays
            try:
                arr = np.array(preds)
                preds = arr.reshape(-1).astype(float).tolist()
            except Exception:
                pass

        return json.dumps({"predictions": preds}).encode("utf-8"), content_type
    except Exception as e:
        # return a clear error message that will appear in the endpoint response
        err = {"error": "output_handler parsing failed", "exception": str(e)}
        return json.dumps(err).encode("utf-8"), content_type
