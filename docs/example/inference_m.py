import os
import json
import yaml
import torch
import joblib
import numpy as np


# 🔹 Definición del modelo (debe coincidir con entrenamiento)
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=3):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 2)

    def forward(self, x):
        return self.linear(x)


# 🔹 Cargar modelo y config
def model_fn(model_dir):
    # cargar config
    with open(os.path.join(model_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    device = config["inference"]["device"]

    # cargar modelo
    model = SimpleModel()
    model.load_state_dict(torch.load(
        os.path.join(model_dir, config["paths"]["model_file"]),
        map_location=device
    ))
    model.eval()

    # cargar scaler
    scaler = joblib.load(os.path.join(model_dir, config["paths"]["scaler_file"]))

    return {
        "model": model,
        "scaler": scaler,
        "config": config
    }


# 🔹 Input
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["inputs"])
    else:
        raise ValueError("Unsupported content type")


# 🔹 Predicción
def predict_fn(input_data, context):
    model = context["model"]
    scaler = context["scaler"]
    config = context["config"]

    # preprocessing
    if config["preprocessing"]["normalize"]:
        input_data = scaler.transform(input_data)

    inputs = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)

    probs = torch.softmax(outputs, dim=1).numpy()

    if config["postprocessing"]["return_proba"]:
        return probs.tolist()
    else:
        preds = np.argmax(probs, axis=1)
        return preds.tolist()


# 🔹 Output
def output_fn(prediction, content_type):
    return json.dumps({"predictions": prediction}), content_type