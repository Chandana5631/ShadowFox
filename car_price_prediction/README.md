# Car Price Prediction System

A comprehensive Machine Learning web application for predicting car selling prices based on various features including fuel type, year of manufacture, kilometers driven, transmission type, and more.

## Features

- **Price Prediction**: Get accurate selling price predictions for cars
- **Data Analysis**: Explore dataset statistics and distributions
- **Model Information**: View feature importance and model performance
- **Modern UI**: Beautiful, responsive web interface
- **Real-time Predictions**: Instant price predictions using trained ML model

## Features Used for Prediction

- **Year of Manufacture**: Manufacturing year of the car
- **Present Price**: Current market price of the car (in Lakhs)
- **Kilometers Driven**: Total distance covered by the car
- **Fuel Type**: Petrol, Diesel, or CNG
- **Seller Type**: Dealer or Individual
- **Transmission**: Manual or Automatic
- **Number of Previous Owners**: 0, 1, 2, or 3

## Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Random Forest)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd car_price_prediction

# Or download and extract the project files
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset

The application will automatically attempt to download the dataset from Google Drive. If it fails, you can manually download it:

1. Visit the dataset link: https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view?usp=drive_link
2. Download the CSV file
3. Place it in the `data/` folder as `car_data.csv`

### Step 4: Train the Model

```bash
python model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest model
- Save the trained model to `models/car_price_model.pkl`

### Step 5: Run the Application

```bash
python app.py
```

The web application will be available at: `http://localhost:5000`

## Usage

### Price Prediction

1. Open the web application in your browser
2. Fill in the car details in the form:
   - Year of manufacture
   - Present price (in Lakhs)
   - Kilometers driven
   - Number of previous owners
   - Fuel type
   - Seller type
   - Transmission type
3. Click "Predict Price" to get the estimated selling price

### Data Analysis

- Navigate to the "Data Analysis" tab to view:
  - Dataset statistics
  - Fuel type distribution
  - Transmission distribution
  - Seller type distribution

### Model Information

- Navigate to the "Model Info" tab to view:
  - Feature importance rankings
  - Model performance metrics
  - Model training options

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Get price prediction
- `GET /api/data_analysis`: Get dataset statistics
- `GET /api/model_info`: Get model information
- `POST /train`: Retrain the model
- `GET /load_model`: Load existing model

## Model Performance

The Random Forest model typically achieves:
- **R² Score**: ~0.85-0.90
- **RMSE**: ~1.5-2.0 Lakhs
- **MAE**: ~1.0-1.5 Lakhs

## Project Structure

```
car_price_prediction/
├── app.py                 # Main Flask application
├── model.py              # ML model training and prediction
├── data_downloader.py    # Dataset download utility
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── templates/
│   └── index.html       # Web interface template
├── data/                # Dataset storage
│   └── car_data.csv     # Car dataset
└── models/              # Trained model storage
    └── car_price_model.pkl
```

## Customization

### Adding New Features

To add new features to the prediction model:

1. Update the dataset with new columns
2. Modify the `preprocess_data()` method in `model.py`
3. Update the web form in `templates/index.html`
4. Update the prediction endpoint in `app.py`

### Changing the Model

To use a different ML algorithm:

1. Modify the `train_model()` method in `model.py`
2. Choose from available options:
   - Random Forest (default)
   - Gradient Boosting
   - Linear Regression

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure `car_data.csv` is in the `data/` folder
2. **Model not trained**: Run `python model.py` to train the model
3. **Port already in use**: Change the port in `app.py` or kill the existing process

### Error Messages

- **"Model not trained"**: Train the model using `python model.py`
- **"Dataset not found"**: Download and place the dataset in the correct location
- **"Invalid input"**: Check that all form fields are filled correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error messages
3. Ensure all dependencies are installed
4. Verify the dataset is properly loaded

## Future Enhancements

- [ ] Add more ML algorithms
- [ ] Implement model comparison
- [ ] Add data visualization charts
- [ ] Support for batch predictions
- [ ] Export predictions to CSV
- [ ] User authentication system
- [ ] Prediction history tracking

