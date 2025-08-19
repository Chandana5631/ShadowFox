import pandas as pd
import numpy as np
from model import CarPricePredictor
import json

def batch_predict(cars_data):
    """
    Predict prices for multiple cars
    
    Args:
        cars_data: List of dictionaries containing car features
    
    Returns:
        List of predictions with input features
    """
    
    # Load the trained model
    predictor = CarPricePredictor()
    if not predictor.load_model():
        print("❌ Model not found. Please train the model first using: python model.py")
        return None
    
    predictions = []
    
    for i, car in enumerate(cars_data):
        try:
            # Make prediction
            predicted_price = predictor.predict_price(car)
            
            # Add to results
            result = {
                'car_id': i + 1,
                'features': car,
                'predicted_price': round(predicted_price, 2)
            }
            predictions.append(result)
            
            print(f"Car {i+1}: ₹{predicted_price:.2f} Lakhs")
            
        except Exception as e:
            print(f"❌ Error predicting car {i+1}: {e}")
            predictions.append({
                'car_id': i + 1,
                'features': car,
                'error': str(e)
            })
    
    return predictions

def load_cars_from_csv(file_path):
    """Load car data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        cars = []
        
        for _, row in df.iterrows():
            car = {
                'Year': int(row['Year']),
                'Present_Price': float(row['Present_Price']),
                'Kms_Driven': int(row['Kms_Driven']),
                'Fuel_Type': row['Fuel_Type'],
                'Seller_Type': row['Seller_Type'],
                'Transmission': row['Transmission'],
                'Owner': int(row['Owner'])
            }
            cars.append(car)
        
        return cars
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None

def save_predictions(predictions, output_file='batch_predictions.json'):
    """Save predictions to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✅ Predictions saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")

def main():
    """Main function for batch prediction"""
    print("🚗 Car Price Batch Prediction")
    print("=" * 40)
    
    # Example cars for prediction
    sample_cars = [
        {
            'Year': 2018,
            'Present_Price': 8.5,
            'Kms_Driven': 45000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Individual',
            'Transmission': 'Manual',
            'Owner': 0
        },
        {
            'Year': 2020,
            'Present_Price': 12.0,
            'Kms_Driven': 25000,
            'Fuel_Type': 'Diesel',
            'Seller_Type': 'Dealer',
            'Transmission': 'Automatic',
            'Owner': 1
        },
        {
            'Year': 2015,
            'Present_Price': 5.5,
            'Kms_Driven': 80000,
            'Fuel_Type': 'Petrol',
            'Seller_Type': 'Individual',
            'Transmission': 'Manual',
            'Owner': 2
        }
    ]
    
    print("📋 Sample cars for prediction:")
    for i, car in enumerate(sample_cars):
        print(f"Car {i+1}: {car['Year']} {car['Fuel_Type']} {car['Transmission']} - {car['Kms_Driven']} km")
    
    print("\n🔮 Making predictions...")
    predictions = batch_predict(sample_cars)
    
    if predictions:
        print("\n📊 Prediction Summary:")
        total_predicted = sum(p['predicted_price'] for p in predictions if 'predicted_price' in p)
        avg_predicted = total_predicted / len([p for p in predictions if 'predicted_price' in p])
        
        print(f"Total cars: {len(predictions)}")
        print(f"Average predicted price: ₹{avg_predicted:.2f} Lakhs")
        print(f"Total predicted value: ₹{total_predicted:.2f} Lakhs")
        
        # Save predictions
        save_predictions(predictions)
        
        # Display detailed results
        print("\n📋 Detailed Results:")
        for pred in predictions:
            if 'predicted_price' in pred:
                car = pred['features']
                print(f"Car {pred['car_id']}: ₹{pred['predicted_price']:.2f}L "
                      f"({car['Year']} {car['Fuel_Type']} {car['Transmission']})")
            else:
                print(f"Car {pred['car_id']}: Error - {pred['error']}")

if __name__ == "__main__":
    main()

