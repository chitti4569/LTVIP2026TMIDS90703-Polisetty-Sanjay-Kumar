import numpy as np
import os
import pickle
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img

app = Flask(__name__)

model = tf.keras.models.load_model("dogbreed.h5")

with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

class_names = list(class_indices.keys())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/output", methods=["POST"])
def output():
    file = request.files["file"]

    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    image_array = np.array(img) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    pred_index = np.argmax(prediction)

    predicted_class = class_names[pred_index]

    return render_template("output.html",
                           predict=predicted_class,
                           image_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
