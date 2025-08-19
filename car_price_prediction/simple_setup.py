#!/usr/bin/env python3
"""
Simple Setup Script for Car Price Prediction System
Uses minimal dependencies to avoid conflicts
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚗 Car Price Prediction System - Simple Setup")
    print("=" * 50)
    print()
    
    # Step 1: Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Step 2: Install minimal dependencies
    print("📦 Installing minimal dependencies...")
    if not run_command("pip install -r requirements_minimal.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        print("Trying individual packages...")
        
        packages = ["pandas", "numpy", "scikit-learn", "flask", "flask-cors", "requests", "joblib"]
        for package in packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"❌ Failed to install {package}")
                sys.exit(1)
    
    # Step 3: Create directories
    directories = ["data", "models", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Step 4: Generate sample data
    print("📊 Generating sample dataset...")
    if not run_command("python generate_sample_data.py", "Generating sample data"):
        print("❌ Failed to generate sample data")
        sys.exit(1)
    
    # Step 5: Train model
    print("🤖 Training model...")
    if not run_command("python model.py", "Training model"):
        print("❌ Failed to train model")
        sys.exit(1)
    
    # Step 6: Test application
    print("🧪 Testing application...")
    try:
        from app import app
        print("✅ Application test passed")
    except Exception as e:
        print(f"❌ Application test failed: {e}")
        sys.exit(1)
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("To run the application:")
    print("  python app.py")
    print()
    print("Then open your browser and go to:")
    print("  http://localhost:5000")

if __name__ == "__main__":
    main()
