from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import urllib.request
from werkzeug.utils import secure_filename
import logging

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/predicted'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
MODEL_PATH = "/Users/user/Documents/I5/AI/FinalProject2/vehicle-image-classifier/resnet50_vehicle_classifier1.pth"
CLASS_NAMES = ['airplane', 'bicycles', 'cars', 'motorbikes', 'ships']
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for determining "Unknown Class"

# Configure Flask app
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH
)

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize ResNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Load the model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # Load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    logger.info("ResNet model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ResNet model: {str(e)}")
    raise

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for the ResNet model."""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_image(image_path):
    """Predict the class of an image using the ResNet model."""
    try:
        # Preprocess the image
        image_tensor = preprocess_image(image_path)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_idx = outputs.max(1)
            confidence = torch.softmax(outputs, dim=1)[0, predicted_idx].item()

            # Determine the predicted class name
            predicted_class_name = "Unknown Class"
            if confidence >= CONFIDENCE_THRESHOLD and predicted_idx.item() < len(CLASS_NAMES):
                predicted_class_name = CLASS_NAMES[predicted_idx.item()]

        # Save the predicted image without annotation for simplicity
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
        Image.open(image_path).save(output_path)  # Save a copy of the original image

        # Log the result
        logger.info(f"Predicted class: {predicted_class_name}, Confidence: {confidence:.2f}")

        return {
            "class": predicted_class_name,
            "confidence": confidence,
            "output_image": f"/static/predicted/{os.path.basename(image_path)}"
        }, None
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}, None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from various sources."""
    try:
        if 'file' in request.files:
            return handle_file_upload(request.files['file'])
        elif 'url' in request.form:
            return handle_url_input(request.form['url'])
        elif 'path' in request.form:
            return handle_path_input(request.form['path'])
        else:
            return jsonify({"error": "No file, URL, or path provided"}), 400
    except Exception as e:
        logger.error(f"Prediction request error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def handle_file_upload(file):
    """Handle file upload prediction requests."""
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result, error = predict_image(filepath)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

def handle_url_input(url):
    """Handle URL-based prediction requests."""
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    filepath = download_image(url)
    result, error = predict_image(filepath)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

def handle_path_input(image_path):
    """Handle local path prediction requests."""
    if not os.path.exists(image_path):
        return jsonify({"error": "File path does not exist"}), 400
    if not allowed_file(image_path):
        return jsonify({"error": "File type not allowed"}), 400

    result, error = predict_image(image_path)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)

def download_image(url):
    """Download image from URL and save it temporarily."""
    try:
        filename = secure_filename(os.path.basename(url))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return jsonify({"error": "File size exceeded the limit (16MB)"}), 413

if __name__ == '__main__':
    app.run(debug=True)
