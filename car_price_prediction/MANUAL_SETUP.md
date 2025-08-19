# Manual Setup Guide

If you encounter dependency conflicts, follow this step-by-step manual setup:

## Step 1: Install Dependencies One by One

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install flask
pip install flask-cors
pip install requests
pip install joblib
```

## Step 2: Create Directories

```bash
mkdir data
mkdir models
mkdir templates
```

## Step 3: Generate Sample Data

```bash
python generate_sample_data.py
```

## Step 4: Train the Model

```bash
python model.py
```

## Step 5: Run the Application

```bash
python app.py
```

## Alternative: Use Virtual Environment

If you continue to have issues, create a virtual environment:

```bash
# Create virtual environment
python -m venv car_prediction_env

# Activate virtual environment
# On Windows:
car_prediction_env\Scripts\activate
# On macOS/Linux:
source car_prediction_env/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn flask flask-cors requests joblib

# Run the application
python app.py
```

## Troubleshooting

### If you get "Module not found" errors:
1. Make sure you're in the correct directory
2. Check that all dependencies are installed: `pip list`
3. Try installing missing packages individually

### If you get permission errors:
1. On Windows, run PowerShell as Administrator
2. On macOS/Linux, use `sudo pip install` (not recommended) or use virtual environment

### If the web interface doesn't load:
1. Check that Flask is running on http://localhost:5000
2. Make sure no other application is using port 5000
3. Try a different port by modifying `app.py`

## Quick Test

To test if everything is working:

```bash
python -c "
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from flask import Flask
print('✅ All dependencies installed successfully!')
"
```

## Access the Application

Once everything is set up:
1. Open your web browser
2. Go to: http://localhost:5000
3. You should see the Car Price Prediction interface
