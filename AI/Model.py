import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os
import base64


app = Flask(__name__)
CORS(app)

STATIC_FOLDER = 'static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Check if the 'static' folder exists, and create it if not
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome -- It's our Graduation AI Model"

@app.route('/', methods=['POST'])
def generateImage():
    if 'image_data' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image_data']

    # Save the uploaded file to a temporary location
    temp_path = save_temporary_file(image_file)

    # Load the model
    model = load_model('chexnet_model.h5')

    # Read the original image
    original_img = cv2.imread(temp_path)

    if original_img is None:
        print(f"Error: Unable to read the image at {temp_path}")
        return jsonify({'error': 'Failed to read the image file'})

    # Preprocess the image for the model
    img = cv2.resize(original_img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = model.predict(img_array)
    class_labels = ['Normal', 'Pneumonia']
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_labels[tf.argmax(score)]
    confidence = 100 * tf.reduce_max(score)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Get the last convolutional layer
    last_conv_layer = model.get_layer('conv5_block16_concat')

    # Create a model that outputs the last convolutional layer and the final model output
    cam_model = tf.keras.Model(inputs=model.input, outputs=(last_conv_layer.output, model.output))

    # Use GradientTape to compute gradients
    x = img_array  # Use the preprocessed image as input
    with tf.GradientTape() as tape:
        last_conv_output, preds = cam_model(x)
        class_output = preds[:, tf.argmax(score)]

    # Compute gradients and pooled gradients
    grads = tape.gradient(class_output, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply the last convolutional layer's output with pooled gradients to get the heatmap
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    # Resize heatmap to match the original image
    heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))

    # Apply the heatmap on the original image
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    image_name = f'superimposed_{time.time()}.png'
    static_path = os.path.join(app.config['STATIC_FOLDER'], image_name)
    cv2.imwrite(static_path, superimposed_img)
    # Print the list of files in the static folder
    print(os.listdir(app.config['STATIC_FOLDER']))
    image_url = f'http://localhost:5000/{image_name}'
    response_data = {'image_url': image_url, "Predicted class": predicted_class}

    # Remove the temporary file
    os.remove(temp_path)

    return jsonify(response_data) 

@app.route('/<path:filename>')
def serve_static(filename):
    if filename == 'latest':
        # Get the latest modified file in the static folder
        files = glob.glob(os.path.join(app.config['STATIC_FOLDER'], '*.png'))
        latest_file = max(files, key=os.path.getmtime)
        return send_from_directory(app.config['STATIC_FOLDER'], os.path.basename(latest_file))
    else:
        return send_from_directory(app.config['STATIC_FOLDER'], filename)

def save_temporary_file(file):
    _, temp_path = tempfile.mkstemp()
    file.save(temp_path)
    return temp_path

if __name__ == '__main__':
    app.run(host='0.0.0.0')