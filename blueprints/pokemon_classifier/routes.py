from flask import Blueprint, request, jsonify, current_app as app
from blueprints.pokemon_classifier.model import PokemonClassifier
from werkzeug.utils import secure_filename
import os
import requests
from urllib.parse import urlparse

# Flask Blueprint for the API
pokemon_classifier = Blueprint('pokemon_classifier', __name__)

# Initialize the classifier [change: pass it in from the main app?]
classifier = PokemonClassifier(
    model_path="models/pokemon_v1.pth",
    class_mapping_path="models/class_mapping.json"
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@pokemon_classifier.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Classifier is running'})

@pokemon_classifier.route('/url_test', methods=['GET'])
def testing():
    # Test picture of Pikachu
    url = "https://i.pinimg.com/736x/bf/95/34/bf953419d76bf747cba69b55e6e03957.jpg"
    response = requests.get(url)
    response.raise_for_status()

    parsed_url = urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    
    filename = secure_filename(original_filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    prediction = classifier.predict(filepath, return_top_k=3)
    os.remove(filepath)
    return jsonify(prediction)

@pokemon_classifier.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type or no file selected'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        prediction = classifier.predict(filepath, return_top_k=3)
        os.remove(filepath)
    
        return jsonify(prediction)
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500