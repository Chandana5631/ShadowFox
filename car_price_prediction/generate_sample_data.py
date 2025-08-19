import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_sample_data(n_samples=1000):
    """Generate sample car data for testing"""
    
    np.random.seed(42)  # For reproducible results
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data
    data = {
        'Car_Name': [f'Car_{i+1}' for i in range(n_samples)],
        'Year': np.random.randint(2000, 2024, n_samples),
        'Selling_Price': np.random.uniform(0.5, 25.0, n_samples),
        'Present_Price': np.random.uniform(1.0, 30.0, n_samples),
        'Kms_Driven': np.random.randint(1000, 150000, n_samples),
        'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'CNG'], n_samples, p=[0.5, 0.4, 0.1]),
        'Seller_Type': np.random.choice(['Dealer', 'Individual'], n_samples, p=[0.6, 0.4]),
        'Transmission': np.random.choice(['Manual', 'Automatic'], n_samples, p=[0.7, 0.3]),
        'Owner': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.3, 0.08, 0.02])
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Adjust selling price based on features (more realistic)
    # Newer cars with lower mileage should have higher selling prices
    df['Selling_Price'] = (
        df['Present_Price'] * 0.8 +  # Base price
        (df['Year'] - 2000) * 0.1 +  # Year factor
        (150000 - df['Kms_Driven']) / 10000 +  # Mileage factor
        np.where(df['Fuel_Type'] == 'Diesel', 1.0, 0.0) +  # Diesel premium
        np.where(df['Transmission'] == 'Automatic', 1.5, 0.0) +  # Automatic premium
        np.where(df['Seller_Type'] == 'Dealer', 0.5, 0.0) +  # Dealer premium
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Ensure selling price is positive and reasonable
    df['Selling_Price'] = np.maximum(df['Selling_Price'], 0.5)
    df['Selling_Price'] = np.minimum(df['Selling_Price'], df['Present_Price'] * 1.2)
    
    # Round prices to 2 decimal places
    df['Selling_Price'] = df['Selling_Price'].round(2)
    df['Present_Price'] = df['Present_Price'].round(2)
    
    # Save to CSV
    output_path = 'data/car_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated sample dataset with {n_samples} records")
    print(f"📁 Saved to: {output_path}")
    print()
    print("📊 Dataset Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Price range: ₹{df['Selling_Price'].min():.2f}L - ₹{df['Selling_Price'].max():.2f}L")
    print(f"   Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Average price: ₹{df['Selling_Price'].mean():.2f}L")
    print()
    print("🔍 Feature distributions:")
    print(f"   Fuel types: {df['Fuel_Type'].value_counts().to_dict()}")
    print(f"   Transmission: {df['Transmission'].value_counts().to_dict()}")
    print(f"   Seller types: {df['Seller_Type'].value_counts().to_dict()}")
    print(f"   Owners: {df['Owner'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    print("🚗 Generating Sample Car Dataset")
    print("=" * 40)
    generate_sample_data()

