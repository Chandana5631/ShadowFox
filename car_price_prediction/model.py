import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Selling_Price'
        
    def load_data(self, file_path='data/car_data.csv'):
        """Load and preprocess the car dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Dataset not found at {file_path}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data = data.dropna()
        
        # Convert categorical variables
        categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Define feature columns (excluding target and non-numeric columns)
        exclude_columns = [self.target_column, 'Car_Name']
        self.feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Separate features and target
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Scale numerical features
        numerical_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
        if all(col in X.columns for col in numerical_columns):
            X_scaled = X.copy()
            X_scaled[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
            X = X_scaled
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the ML model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError("Model type must be 'random_forest', 'gradient_boosting', or 'linear'")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance ({model_type}):")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'test_size': len(X_test)
        }
    
    def predict_price(self, features):
        """Predict car price based on input features"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert input features to DataFrame
        input_df = pd.DataFrame([features])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # Scale numerical features
        numerical_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
        if all(col in input_df.columns for col in numerical_columns):
            input_df_scaled = input_df.copy()
            input_df_scaled[numerical_columns] = self.scaler.transform(input_df[numerical_columns])
            input_df = input_df_scaled
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        return prediction
    
    def save_model(self, filepath='models/car_price_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/car_price_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file not found at {filepath}")
            return False
    
    def get_feature_importance(self):
        """Get feature importance if using tree-based model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return None

def train_and_save_model():
    """Train the model and save it"""
    predictor = CarPricePredictor()
    
    # Load data
    df = predictor.load_data()
    if df is None:
        return None
    
    # Preprocess data
    X, y = predictor.preprocess_data(df)
    
    # Train model
    performance = predictor.train_model(X, y, model_type='random_forest')
    
    # Save model
    predictor.save_model()
    
    return predictor, performance

if __name__ == "__main__":
    train_and_save_model()

