from flask import Flask, jsonify
from flask_cors import CORS
from blueprints.pokemon_classifier import pokemon_classifier

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Configuration
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Health check
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'message': 'ML Pokemon API is running'
        }), 200

    # Register Blueprints
    app.register_blueprint(pokemon_classifier, url_prefix='/classifier')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5050)