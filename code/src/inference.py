import json

def _parse_input(data):
    """Parse input data to JSON"""
    if isinstance(data, dict):
        return data
    if isinstance(data, (bytes, bytearray)):
        s = data.decode("utf-8")
        return json.loads(s)
    if hasattr(data, "read"):
        raw = data.read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    if isinstance(data, str):
        return json.loads(data)
    raise ValueError("Unsupported input type")

def input_handler(data, context):
    """Simple handler that only accepts pre-processed instances"""
    js = _parse_input(data)
    
    if "instances" not in js:
        raise ValueError("Request JSON must contain 'instances' key with pre-processed data")
    
    return json.dumps({"instances": js["instances"]}), "application/json"

def output_handler(response, accept):
    """Handle TensorFlow Serving response"""
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    
    try:
        parsed = json.loads(response)
        return json.dumps(parsed), "application/json"
    except Exception:
        return response, accept