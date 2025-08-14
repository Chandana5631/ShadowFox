from flask import Flask, render_template, request, send_file
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import io
import csv

app = Flask(__name__)

# Load ResNet50 pre-trained model
model = ResNet50(weights="imagenet")

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Preprocess uploaded image
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))   # ResNet expects 224x224
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=5)[0]  # Top 5 predictions

    # Save uploaded image temporarily for display
    img.save("static/uploaded_image.jpg")

    results = [(label, prob * 100) for (_, label, prob) in decoded_preds]
    return render_template("index.html", predictions=results, image="static/uploaded_image.jpg")

# CSV download route
@app.route("/download_report", methods=["POST"])
def download_report():
    predictions = request.form.getlist("predictions[]")

    # Write CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Label", "Confidence (%)"])

    for pred in predictions:
        label, conf = pred.split(":")
        writer.writerow([label.strip(), conf.strip()])

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="prediction_report.csv")

if __name__ == "__main__":
    app.run(debug=True)
