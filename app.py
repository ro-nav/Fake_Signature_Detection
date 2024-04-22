from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import io

app = Flask(__name__)
CORS(app)
model = load_model("model\signature-detection-without-datagen.keras")

def preprocess_image(file_storage):
    # Convert the FileStorage object to a file-like object
    file_like = io.BytesIO(file_storage.read())

    img = image.load_img(file_like, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the image_data is in the request as a file
        image_file = request.files['image_data']
        print(image_file)
        processed_image = preprocess_image(image_file)

        # Make prediction
        result = model.predict(processed_image)
        
        # For testing purposes, return a simple response with 'application/json' content type

        response = jsonify({'result': result.tolist()})
        response.headers.add('Content-Type', 'application/json')
        return response
    except Exception as e:
        return jsonify({'result': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
