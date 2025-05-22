from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variables for model components
model = None
scaler = None
encoders = None
feature_names = None

def load_model_components():
    """Load all model components at startup"""
    global model, scaler, encoders, feature_names
    
    try:
        # Load model
        model = joblib.load('housingModel.joblib')
        logger.info("✅ Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load('scaler.joblib')
        logger.info("✅ Scaler loaded successfully")
        
        # Load encoders
        encoders = joblib.load('encoders.joblib')
        logger.info("✅ Encoders loaded successfully")
        
        # Load feature names
        feature_names = joblib.load('feature_names.joblib')
        logger.info("✅ Feature names loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model components: {str(e)}")
        return False

def prepare_features(input_data):
    """Prepare input data for prediction"""
    try:
        # Create a DataFrame with the same structure as training data
        df = pd.DataFrame([input_data])
        
        # Apply label encoders to categorical columns
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    # Handle unknown categories by using the most frequent class
                    logger.warning(f"Unknown category in {col}, using default encoding")
                    df[col] = 0
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training order
        df = df[feature_names]
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        return scaled_features[0]
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "🏠 Housing Price Prediction API",
        "version": "1.0",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make price predictions",
            "/health": "GET - Check API health",
            "/info": "GET - Get model information"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": encoders is not None
    })

@app.route('/info')
def model_info():
    """Get model information"""
    if not all([model, scaler, encoders, feature_names]):
        return jsonify({"error": "Model components not loaded"}), 500
    
    return jsonify({
        "model_type": type(model).__name__,
        "features": feature_names,
        "categorical_features": list(encoders.keys()),
        "num_features": len(feature_names),
        "scaler_type": type(scaler).__name__
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make price predictions"""
    try:
        # Check if model is loaded
        if not all([model, scaler, encoders]):
            return jsonify({
                "success": False,
                "error": "Model components not loaded"
            }), 500
        
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        logger.info(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                          'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
                          'parking', 'prefarea', 'furnishingstatus']
        
        # Convert numeric fields
        numeric_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    return jsonify({
                        "success": False,
                        "error": f"Invalid numeric value for {field}"
                    }), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Log successful prediction
        logger.info(f"Prediction made: {prediction}")
        
        return jsonify({
            "success": True,
            "predicted_price": round(float(prediction), 2),
            "input_data": data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict/simple', methods=['POST'])
def predict_simple():
    """Simplified prediction endpoint with basic features"""
    try:
        # Check if model is loaded
        if not all([model, scaler, encoders]):
            return jsonify({
                "success": False,
                "error": "Model components not loaded"
            }), 500
        
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Simple mapping for basic inputs
        simple_data = {
            'area': float(data.get('area', 1000)),
            'bedrooms': int(data.get('bedrooms', 3)),
            'bathrooms': int(data.get('bathrooms', 2)),
            'stories': int(data.get('stories', 1)),
            'mainroad': data.get('mainroad', 'yes'),
            'guestroom': data.get('guestroom', 'no'),
            'basement': data.get('basement', 'no'),
            'hotwaterheating': data.get('hotwaterheating', 'no'),
            'airconditioning': data.get('airconditioning', 'no'),
            'parking': int(data.get('parking', 1)),
            'prefarea': data.get('prefarea', 'no'),
            'furnishingstatus': data.get('furnishingstatus', 'unfurnished')
        }
        
        logger.info(f"Simple prediction request: {simple_data}")
        
        # Prepare features
        features = prepare_features(simple_data)
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return jsonify({
            "success": True,
            "predicted_price": round(float(prediction), 2),
            "price_per_sqft": round(float(prediction) / simple_data['area'], 2),
            "input_summary": {
                "area": simple_data['area'],
                "bedrooms": simple_data['bedrooms'],
                "bathrooms": simple_data['bathrooms'],
                "location_quality": "Good" if simple_data['prefarea'] == 'yes' else "Standard"
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Simple prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("="*50)
    print("🏠 HOUSING PRICE PREDICTION API")
    print("="*50)
    
    # Load model components
    if load_model_components():
        print("✅ All components loaded successfully!")
        print("\n📡 API Endpoints:")
        print("- GET  /          : API information")
        print("- GET  /health    : Health check")
        print("- GET  /info      : Model information")
        print("- POST /predict   : Full prediction")
        print("- POST /predict/simple : Simple prediction")
        
        print(f"\n🚀 Starting server on http://localhost:5000")
        print("="*50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model components. Please check your files.")
        print("Required files:")
        print("- housingModel.joblib")
        print("- scaler.joblib") 
        print("- encoders.joblib")
        print("- feature_names.joblib")