
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model/Updated-Xception-diabetic-retinopathy.h5")

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes = [
"No Diabetic Retinopathy",
"Mild Diabetic Retinopathy",
"Moderate Diabetic Retinopathy",
"Severe Diabetic Retinopathy",
"Proliferative Diabetic Retinopathy"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    return render_template("prediction.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
