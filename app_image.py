from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
import json
import traceback
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

interpreter = tflite.Interpreter(model_path="car_damage_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 160
CLASS_NAMES = ["damaged", "whole"]

def preprocess_input(x):
    x = x.astype(np.float32)
    x = (x / 127.5) - 1.0
    return x

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Car Damage Detection API (TFLite) is running"})

@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        raw_data = request.get_data(as_text=True)
        decoder = json.JSONDecoder()
        data, idx = decoder.raw_decode(raw_data)

        if not data or 'image_base64' not in data:
            return jsonify({"error": "Provide 'image_base64' in JSON body"}), 400

        base64_str = data['image_base64']
        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += '=' * (4 - missing_padding)

        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

        return jsonify({
            "prediction_label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)