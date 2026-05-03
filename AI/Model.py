import time
import glob
import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Use /tmp so it works on Railway, Render, and any read-only filesystem host
STATIC_FOLDER = '/tmp/static'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the model once at startup (not on every request)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'chexnet_model.h5')
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

CLASS_LABELS = ['Normal', 'Pneumonia']


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome -- It's our Graduation AI Model"


@app.route('/', methods=['POST'])
def generate_image():
    if 'image_data' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image_data']
    temp_path = None

    try:
        # Save upload to a temp file
        temp_path = save_temporary_file(image_file)

        # Read original image
        original_img = cv2.imread(temp_path)
        if original_img is None:
            return jsonify({'error': 'Failed to read the image file'}), 400

        # Pre-process for model
        img = cv2.resize(original_img, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = CLASS_LABELS[tf.argmax(score).numpy()]
        confidence = float(100 * tf.reduce_max(score))

        print(f"Predicted class : {predicted_class}")
        print(f"Confidence      : {confidence:.2f}%")

        # ── Grad-CAM heatmap ──────────────────────────────────────────────────
        last_conv_layer = model.get_layer('conv5_block16_concat')
        cam_model = tf.keras.Model(
            inputs=model.input,
            outputs=(last_conv_layer.output, model.output)
        )

        with tf.GradientTape() as tape:
            last_conv_output, preds = cam_model(img_array)
            class_output = preds[:, tf.argmax(score)]

        grads = tape.gradient(class_output, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_output = last_conv_output[0]
        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        # Resize & overlay heatmap on original image
        heatmap_np = cv2.resize(
            heatmap.numpy(),
            (original_img.shape[1], original_img.shape[0])
        )
        heatmap_np = (heatmap_np * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        # Save result
        image_name = f'superimposed_{time.time()}.png'
        static_path = os.path.join(STATIC_FOLDER, image_name)
        cv2.imwrite(static_path, superimposed)

        return jsonify({
            'image_url': f'/static/{image_name}',
            'predicted_class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/static/<path:filename>')
def serve_static(filename):
    if filename == 'latest':
        files = glob.glob(os.path.join(STATIC_FOLDER, '*.png'))
        if not files:
            return jsonify({'error': 'No images available'}), 404
        latest_file = max(files, key=os.path.getmtime)
        return send_from_directory(STATIC_FOLDER, os.path.basename(latest_file))
    return send_from_directory(STATIC_FOLDER, filename)


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_temporary_file(file):
    suffix = os.path.splitext(file.filename)[-1] or '.tmp'
    _, temp_path = tempfile.mkstemp(suffix=suffix)
    file.save(temp_path)
    return temp_path


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)