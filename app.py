
import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from flask import Flask, jsonify, request, render_template, abort
from flask_cors import CORS
import mediapipe as mp

# Project paths
BASE_DIR = Path("/home/aahana/SANKET_Backend")
CHECKPOINT_PATH = BASE_DIR / "best_resnet18_combined.pth"
HAND_MODEL_PATH = BASE_DIR / "models/hand_landmarker.task"
FRONTEND_DIR = BASE_DIR / "frontend/SANKET_frontend"
TEMPLATE_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"

# Flask app and device
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained ResNet-18
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)
classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Initialize MediaPipe hand detector
if not HAND_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Hand detector model not found: {HAND_MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)
hand_detector = HandLandmarker.create_from_options(hand_options)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/<page>")
def frontend_page(page):
    if not page.endswith(".html") or not (TEMPLATE_DIR / page).is_file():
        abort(404)
    return render_template(page)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "ResNet-18",
        "classes": len(classes),
        "device": str(device),
        "hand_detector": "ready"
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")

        if not image_data:
            return jsonify({
                "error": "No image provided"
            }), 400

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detect the hand
        rgb = np.ascontiguousarray(np.asarray(image))
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = hand_detector.detect(mp_image)

        if not result.hand_landmarks:
            return jsonify({
                "hand_detected": False,
                "message": "No hand detected"
            })

        # Crop around the detected hand with extra space
        landmarks = result.hand_landmarks[0]
        image_width, image_height = image.size

        xs = [point.x * image_width for point in landmarks]
        ys = [point.y * image_height for point in landmarks]

        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        side = max(right - left, bottom - top) * 1.5
        side = max(side, 32)

        crop = image.crop((
            round(center_x - side / 2),
            round(center_y - side / 2),
            round(center_x + side / 2),
            round(center_y + side / 2)
        ))

        # Run the trained classifier
        input_tensor = transform(crop).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)

        label = classes[predicted_index.item()]
        score = float(confidence.item())

        return jsonify({
            "hand_detected": True,
            "prediction": label,
            "confidence": score
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    print("SANKET backend starting...")
    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print("Website: http://localhost:5000/live.html")
    app.run(host="0.0.0.0", port=5000, debug=False)
