from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('my_weights_new_opt.h5')

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = f"./{file.filename}"
        file.save(file_path)

        image = preprocess_image(file_path)
        result = model.predict(image)

        if result[0][0] > 0.5:
            prediction = "normal"
        else:
            prediction = "cancer"

        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
