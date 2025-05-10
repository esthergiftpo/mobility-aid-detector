
from flask import Flask, render_template, request, jsonify
import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model("best_model.keras")
classes = ['crutches', 'no mobility aids', 'wheelchair', 'whitecane']
threshold = 0.3

def preprocess_image(data_url):
    img_str = re.search(r'base64,(.*)', data_url).group(1)
    img_data = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image')
    if image_data:
        img_array = preprocess_image(image_data)
        preds = model.predict(img_array)[0]
        binary_preds = (preds > threshold).astype(int)
        predictions = [classes[i] for i, val in enumerate(binary_preds) if val]
        return jsonify({'predictions': predictions})
    return jsonify({'error': 'No image received'}), 400

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

