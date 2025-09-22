from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from io import BytesIO
import qrcode
import hashlib
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import uuid
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import torch
import piq

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Allowed extensions for photo upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Simple Blockchain Implementation ---
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": str(self.timestamp),
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

blockchain = Blockchain()

# --- QR Code generation ---
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=4,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    import base64
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

# --- AI Model Setup ---
plant_model = hub.load("https://tfhub.dev/google/plant_village/efficientnet_b0/classifier/1")
input_shape = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_herb_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # BRISQUE score using piq
        score = piq.brisque(img_tensor, data_range=1.0).item()
        print(f"BRISQUE score: {score}")
        if score > 50:
            return "Improper Image"

        # Plant Classification
        img = img.resize(input_shape)
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = plant_model(x)
        predicted_class = np.argmax(preds, axis=1)[0]

        return "Good" if predicted_class is not None else "Bad"

    except Exception as e:
        print(f"Error in classification: {e}")
        return "Improper Image"

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', blockchain_chain=blockchain.chain)

@app.route('/add_record', methods=['POST'])
def add_record():
    species = request.form.get("species")
    collector_id = request.form.get("collector_id")
    location = request.form.get("location")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    timestamp = request.form.get("timestamp")
    photo = request.files.get('photo')

    if not all([species, collector_id, location, latitude, longitude, timestamp]):
        flash("Please fill in all fields.", "error")
        return redirect(url_for('index'))

    filename = None
    if photo and allowed_file(photo.filename):
        ext = photo.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(photo_path)
        quality_result = classify_herb_image(photo_path)
    else:
        quality_result = "No Photo Provided"

    collection_event = {
        "resourceType": "CollectionEvent",
        "collectedBy": collector_id,
        "collectedAt": timestamp,
        "location": {
            "name": location,
            "latitude": latitude,
            "longitude": longitude
        }
    }

    quality_test = {
        "resourceType": "QualityTest",
        "testType": "AI Image Classification",
        "result": quality_result,
        "testedAt": datetime.now().isoformat()
    }

    processing_step = {
        "resourceType": "ProcessingStep",
        "description": "Initial collection and quality check",
        "timestamp": datetime.now().isoformat()
    }

    record_data = {
        'species': species,
        'collectionEvent': collection_event,
        'qualityTest': quality_test,
        'processingStep': processing_step,
        'photoFilename': filename
    }

    new_block = Block(
        index=len(blockchain.chain),
        timestamp=datetime.now(),
        data=record_data,
        previous_hash=blockchain.get_latest_block().hash
    )
    blockchain.add_block(new_block)

    flash(f"Record added successfully! Quality: {quality_result}", "success")
    return redirect(url_for('index'))

@app.route('/generate_qr', methods=['POST'])
def generate_qr():
    json_data = request.get_json()
    if not json_data or "data" not in json_data:
        return jsonify({"error": "Missing data"}), 400
    try:
        base64_img = generate_qr_code(json_data["data"])
        return jsonify({'image_data': base64_img})
    except Exception as e:
        print(f"Error generating QR code: {e}")
        return jsonify({"error": "Failed to generate QR code"}), 500

@app.route('/geocode', methods=['POST'])
def geocode():
    data = request.get_json()
    location = data.get("location")
    if not location:
        return jsonify({"error": "Missing location"}), 400
    try:
        with open("opencage_key.txt", "r") as f:
            API_KEY = f.read().strip()
    except Exception as e:
        return jsonify({"error": f"Could not read API key file: {e}"}), 500

    url = "https://api.opencagedata.com/geocode/v1/json"
    response = requests.get(url, params={"q": location, "key": API_KEY, "limit": 1})

    if response.status_code != 200:
        return jsonify({"error": "Geocoding API request failed"}), 500

    results = response.json().get("results")
    if not results:
        return jsonify({"error": "Could not geocode location"}), 400

    geometry = results[0]["geometry"]
    return jsonify({"lat": geometry["lat"], "lng": geometry["lng"]})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
