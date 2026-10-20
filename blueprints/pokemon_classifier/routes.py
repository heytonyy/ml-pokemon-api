from flask import Blueprint, request, jsonify
from blueprints.pokemon_classifier.model import PokemonClassifier
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import logging
import requests

logger = logging.getLogger(__name__)

# Flask Blueprint for the API
pokemon_classifier = Blueprint('pokemon_classifier', __name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/pokemon_v1.pth')
CLASS_MAPPING_PATH = os.environ.get('CLASS_MAPPING_PATH', 'models/class_mapping.json')
# Hugging Face model repo holding the weights, e.g. "heytonyy/pokemon-classifier".
# Used to download the files when they aren't on disk (they're too big for git).
HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO')


def _ensure_local(path: str) -> str:
    if os.path.exists(path) or not HF_MODEL_REPO:
        return path
    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading {os.path.basename(path)} from {HF_MODEL_REPO}")
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=os.path.basename(path))


# Initialize the classifier [change: pass it in from the main app?]
classifier = PokemonClassifier(
    model_path=_ensure_local(MODEL_PATH),
    class_mapping_path=_ensure_local(CLASS_MAPPING_PATH)
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@pokemon_classifier.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Classifier is running'})

@pokemon_classifier.route('/url_test', methods=['GET'])
def testing():
    # Test picture of Pikachu
    url = "https://i.pinimg.com/736x/bf/95/34/bf953419d76bf747cba69b55e6e03957.jpg"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    return jsonify(classifier.predict(image, return_top_k=3))

@pokemon_classifier.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type or no file selected'}), 400

    try:
        image = Image.open(file.stream)
    except UnidentifiedImageError:
        return jsonify({'error': 'File is not a valid image'}), 400

    try:
        return jsonify(classifier.predict(image, return_top_k=3))
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({'error': 'Prediction failed'}), 500
