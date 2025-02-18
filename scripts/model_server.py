# scripts/serve_model.py
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection')

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model loading error: {str(e)}")
    raise

def validate_input(data):
    """Validate input features"""
    required = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                'V28', 'Amount']
    return all(key in data for key in required)

@app.route('/predict', methods=['POST'])
def predict():
    """Fraud prediction endpoint"""
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")
        
        if not validate_input(data):
            return jsonify({'error': 'Invalid input features'}), 400
        
        features = np.array([
            data['Time'],
            *[data[f'V{i}'] for i in range(1, 29)],
            data['Amount']
        ]).reshape(1, -1)
        
        proba = model.predict_proba(features)[0][1]
        prediction = int(proba > 0.5)
        
        response = {
            'prediction': prediction,
            'probability': float(proba),
            'model_version': '1.0.0'
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)