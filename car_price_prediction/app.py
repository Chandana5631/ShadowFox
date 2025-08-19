from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from model import CarPricePredictor

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = CarPricePredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        data = request.get_json()
        
        # Extract features from request
        features = {
            'Year': int(data['year']),
            'Present_Price': float(data['present_price']),
            'Kms_Driven': int(data['kms_driven']),
            'Fuel_Type': data['fuel_type'],
            'Seller_Type': data['seller_type'],
            'Transmission': data['transmission'],
            'Owner': int(data['owner'])
        }
        
        # Make prediction
        predicted_price = predictor.predict_price(features)
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'features': features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model_info')
def model_info():
    """Get model information and performance metrics"""
    try:
        # Get feature importance
        feature_importance = predictor.get_feature_importance()
        
        return jsonify({
            'success': True,
            'feature_importance': feature_importance,
            'feature_columns': predictor.feature_columns
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/data_analysis')
def data_analysis():
    """Get data analysis and visualizations"""
    try:
        # Load data for analysis
        df = predictor.load_data()
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not found'}), 400
        
        # Basic statistics
        stats = {
            'total_cars': len(df),
            'avg_price': round(df['Selling_Price'].mean(), 2),
            'min_price': round(df['Selling_Price'].min(), 2),
            'max_price': round(df['Selling_Price'].max(), 2),
            'avg_year': round(df['Year'].mean(), 0),
            'avg_kms': round(df['Kms_Driven'].mean(), 0)
        }
        
        # Categorical distributions
        fuel_dist = df['Fuel_Type'].value_counts().to_dict()
        transmission_dist = df['Transmission'].value_counts().to_dict()
        seller_dist = df['Seller_Type'].value_counts().to_dict()
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'fuel_distribution': fuel_dist,
            'transmission_distribution': transmission_dist,
            'seller_distribution': seller_dist
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/train', methods=['POST'])
def train_model():
    """Retrain the model"""
    try:
        from model import train_and_save_model
        
        # Train the model
        result = train_and_save_model()
        
        if result is None:
            return jsonify({'success': False, 'error': 'Failed to train model'}), 400
        
        predictor_instance, performance = result
        
        # Update global predictor
        global predictor
        predictor = predictor_instance
        
        return jsonify({
            'success': True,
            'performance': performance,
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/load_model')
def load_model():
    """Load the trained model"""
    try:
        success = predictor.load_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model loaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model file not found'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Try to load existing model
    if not predictor.load_model():
        print("No existing model found. Please train the model first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

