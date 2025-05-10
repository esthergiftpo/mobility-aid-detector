import os
import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("best_model.keras")

# Define your class names
class_names = ["wheelchair", "walking cane", "crutches", "walker", "no aid"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data["image"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    predictions = model.predict(image_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices if predictions[i] > 0.5]

    return jsonify({"predictions": top_classes})