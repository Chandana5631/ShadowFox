#!/usr/bin/env python3
"""
Setup script for Car Price Prediction System
Automates the installation and initial setup process
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚗 Car Price Prediction System Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["data", "models", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def download_dataset():
    """Download the car dataset"""
    print("📥 Downloading dataset...")
    
    # Check if dataset already exists
    if os.path.exists("data/car_data.csv"):
        print("✅ Dataset already exists")
        return True
    
    try:
        # Import the downloader module
        from data_downloader import download_dataset
        result = download_dataset()
        if result is not None:
            print("✅ Dataset downloaded successfully")
            return True
        else:
            print("⚠️  Dataset download failed, generating sample data...")
            return generate_sample_data()
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("⚠️  Generating sample data instead...")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample car data as fallback"""
    try:
        from generate_sample_data import generate_sample_data
        generate_sample_data()
        print("✅ Sample dataset generated successfully")
        return True
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        print("Please download the dataset manually from:")
        print("https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view?usp=drive_link")
        print("And place it in the 'data' folder as 'car_data.csv'")
        return False

def train_model():
    """Train the ML model"""
    print("🤖 Training the model...")
    try:
        from model import train_and_save_model
        result = train_and_save_model()
        if result is not None:
            print("✅ Model trained and saved successfully")
            return True
        else:
            print("❌ Model training failed")
            return False
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_application():
    """Test if the application can start"""
    print("🧪 Testing application...")
    try:
        # Import the app module
        from app import app
        print("✅ Application can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing application: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    print()
    
    # Install dependencies
    install_dependencies()
    print()
    
    # Create directories
    create_directories()
    print()
    
    # Download dataset
    dataset_ok = download_dataset()
    print()
    
    # Train model (only if dataset is available)
    if dataset_ok:
        model_ok = train_model()
        print()
    else:
        model_ok = False
        print("⚠️  Skipping model training (dataset not available)")
        print()
    
    # Test application
    app_ok = test_application()
    print()
    
    # Summary
    print("=" * 60)
    print("📋 Setup Summary:")
    print("=" * 60)
    print(f"✅ Python version: Compatible")
    print(f"✅ Dependencies: Installed")
    print(f"✅ Directories: Created")
    print(f"{'✅' if dataset_ok else '❌'} Dataset: {'Available' if dataset_ok else 'Not available'}")
    print(f"{'✅' if model_ok else '❌'} Model: {'Trained' if model_ok else 'Not trained'}")
    print(f"{'✅' if app_ok else '❌'} Application: {'Ready' if app_ok else 'Not ready'}")
    print()
    
    if dataset_ok and model_ok and app_ok:
        print("🎉 Setup completed successfully!")
        print()
        print("To run the application:")
        print("  python app.py")
        print()
        print("Then open your browser and go to:")
        print("  http://localhost:5000")
    else:
        print("⚠️  Setup completed with some issues:")
        if not dataset_ok:
            print("  - Please download the dataset manually")
        if not model_ok:
            print("  - Please train the model manually: python model.py")
        if not app_ok:
            print("  - Please check the application code")
        print()
        print("After resolving the issues, run:")
        print("  python app.py")

if __name__ == "__main__":
    main()
