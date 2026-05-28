import boto3
import os
import yaml
import torch
import joblib
import json
import numpy as np

BUCKET = "ada-us-east-1-sbx-live-co-risk-data"
PREFIX = "O017332/Model/"

s3 = boto3.client("s3")

def download_file(key, local_path):
    s3.download_file(BUCKET, key, local_path)

def model_fn(model_dir):
    download_file(PREFIX + "model.pth", "/tmp/model.pth")
    download_file(PREFIX + "scaler.pkl", "/tmp/scaler.pkl")
    download_file(PREFIX + "config.yaml", "/tmp/config.yaml")

    with open("/tmp/config.yaml") as f:
        config = yaml.safe_load(f)

    model = torch.load("/tmp/model.pth", map_location="cpu")
    model.eval()

    scaler = joblib.load("/tmp/scaler.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "config": config
    }

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    return np.array(data["inputs"])

def predict_fn(input_data, context):
    model = context["model"]
    scaler = context["scaler"]

    input_data = scaler.transform(input_data)
    inputs = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)

    return outputs.numpy().tolist()

def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction}), accept