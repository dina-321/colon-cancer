from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('my_weights.h5')

def model_predict(image_path, model):
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        result = model_predict(file_path, model)
        os.remove(file_path)  # Remove the file after prediction
        if np.argmax(result) == 0:
            prediction = "Normal"
        else:
            prediction = "Cancer"
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
