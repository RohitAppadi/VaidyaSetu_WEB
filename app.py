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
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Upload settings ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Simple Blockchain ---
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

# --- QR code generation ---
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

# --- PyTorch model placeholder ---
device = torch.device("cpu")
model = resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_herb_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')

        # ---------- Step 1: Image Quality Check ----------
        gray = np.array(img.convert('L'))
        score = ssim(gray, gray) * 100  # dummy placeholder SSIM
        if score < 50:
            return "Poor Image Quality"

        # ---------- Step 2: Plant Classification ----------
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            preds = model(img_t)
        # Simple placeholder: if model outputs something, return Good
        return "Good"
    except Exception as e:
        print(f"Classification error: {e}")
        return "Error"

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        quality_result = "No Photo"

    collection_event = {
        "resourceType": "CollectionEvent",
        "collectedBy": collector_id,
        "collectedAt": timestamp,
        "location": {"name": location, "latitude": latitude, "longitude": longitude}
    }
    quality_test = {
        "resourceType": "QualityTest",
        "testType": "Image Classification",
        "result": quality_result,
        "testedAt": datetime.now().isoformat()
    }
    record_data = {
        'species': species,
        'collectionEvent': collection_event,
        'qualityTest': quality_test,
        'photoFilename': filename
    }

    new_block = Block(len(blockchain.chain), datetime.now(), record_data, blockchain.get_latest_block().hash)
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
        print(f"QR generation error: {e}")
        return jsonify({"error": "Failed"}), 500

@app.route('/geocode', methods=['POST'])
def geocode():
    data = request.get_json()
    location = data.get("location")
    if not location:
        return jsonify({"error": "Missing location"}), 400
    try:
        with open("opencage_key.txt", "r") as f:
            API_KEY = f.read().strip()
        url = "https://api.opencagedata.com/geocode/v1/json"
        response = requests.get(url, params={"q": location, "key": API_KEY, "limit": 1})
        if response.status_code != 200:
            return jsonify({"error": "Geocoding failed"}), 500
        results = response.json().get("results")
        if not results:
            return jsonify({"error": "No results"}), 400
        geometry = results[0]["geometry"]
        return jsonify({"lat": geometry["lat"], "lng": geometry["lng"]})
    except Exception as e:
        print(f"Geocode error: {e}")
        return jsonify({"error": "Failed"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
