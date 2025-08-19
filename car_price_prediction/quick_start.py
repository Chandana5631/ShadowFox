#!/usr/bin/env python3
"""
Quick Start Script for Car Price Prediction System
Automatically sets up and launches the application
"""

import os
import sys
import subprocess
import time
import webbrowser
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

def check_file_exists(file_path):
    """Check if a file exists"""
    return os.path.exists(file_path)

def main():
    """Main quick start function"""
    print("🚗 Car Price Prediction System - Quick Start")
    print("=" * 50)
    print()
    
    # Step 1: Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Step 2: Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Step 3: Create directories
    directories = ["data", "models", "templates"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Step 4: Check for dataset
    if not check_file_exists("data/car_data.csv"):
        print("📥 Dataset not found, attempting to download...")
        if not run_command("python data_downloader.py", "Downloading dataset"):
            print("📊 Generating sample dataset...")
            if not run_command("python generate_sample_data.py", "Generating sample data"):
                print("❌ Failed to get dataset")
                sys.exit(1)
    else:
        print("✅ Dataset found")
    
    # Step 5: Check for trained model
    if not check_file_exists("models/car_price_model.pkl"):
        print("🤖 Training model...")
        if not run_command("python model.py", "Training model"):
            print("❌ Failed to train model")
            sys.exit(1)
    else:
        print("✅ Trained model found")
    
    # Step 6: Test the application
    print("🧪 Testing application...")
    try:
        # Import the app to test if it works
        from app import app
        print("✅ Application test passed")
    except Exception as e:
        print(f"❌ Application test failed: {e}")
        sys.exit(1)
    
    # Step 7: Launch the application
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("🚀 Launching the application...")
    print("📱 The web interface will open in your browser")
    print("🔗 URL: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the application")
    print()
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:5000")
    except:
        print("⚠️  Could not open browser automatically")
        print("Please open: http://localhost:5000")
    
    # Start the Flask application
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()

