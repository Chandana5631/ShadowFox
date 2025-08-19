import requests
import os
import zipfile
import pandas as pd

def download_dataset():
    """Download the car dataset from Google Drive"""
    
    # Google Drive file ID from the URL
    file_id = "1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z"
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open('data/car_data.csv', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Dataset downloaded successfully!")
        
        # Load and display basic info
        df = pd.read_csv('data/car_data.csv')
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the dataset and place it in the 'data' folder as 'car_data.csv'")
        return None

if __name__ == "__main__":
    download_dataset()

