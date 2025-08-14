import io
import json
import numpy as np

def input_handler(data, context):
    """
    Pre-process a whole .npz file (bytes) into a TF-Serving JSON payload.
    For Batch Transform the data arg will be the full object bytes when split_type='None'.
    """
    # read bytes (data can be a file-like or bytes)
    b = data.read() if hasattr(data, "read") else data
    # load npz in-memory
    npz = np.load(io.BytesIO(b), allow_pickle=False)
    # choose first array - change key if you need a specific array e.g. npz['x_test']
    key = list(npz.files)[0]
    arr = npz[key]
    # make sure numbers are serialisable; TF Serving accepts {"instances": [...]}
    return json.dumps({"instances": arr.astype(float).tolist()})

def output_handler(response, context):
    """
    Post-process TFS response. If non-200 raise; otherwise pass bytes back unchanged.
    """
    if response.status_code != 200:
        raise ValueError(response.content.decode("utf-8"))
    # return raw response bytes and the accept header so SDK uploads them as-is
    return response.content, context.accept_header
