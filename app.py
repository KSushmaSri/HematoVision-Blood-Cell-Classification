from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("blood_cell_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert keys to proper mapping
class_indices = {k: int(v) for k, v in class_indices.items()}
class_names = {v: k for k, v in class_indices.items()}

print("Loaded Class Indices:", class_indices)
print("Reversed Class Mapping:", class_names)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 
    return img_array


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        processed_image = preprocess_image(filepath)
        preds = model.predict(processed_image)

        print("Raw Predictions:", preds)

        predicted_index = int(np.argmax(preds))
        confidence = round(float(np.max(preds)) * 100, 2)

        # Safety check
        prediction = class_names.get(predicted_index, "Unknown")

        print("Predicted Index:", predicted_index)
        print("Predicted Label:", prediction)
        print("Confidence:", confidence)

        return render_template("result.html",
                               prediction=prediction,
                               confidence=confidence,
                               filename=file.filename)

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
