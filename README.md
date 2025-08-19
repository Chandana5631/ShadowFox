# ShadowFox - AI & ML Projects Collection 🦊

A comprehensive collection of machine learning and AI projects organized by difficulty level, from beginner to advanced. Each project showcases different technologies and techniques in the field of artificial intelligence and machine learning.

## 📋 Project Overview

This repository contains a curated collection of AI/ML projects developed during the ShadowFox internship program. Projects are categorized by complexity level to help learners progress from basic concepts to advanced implementations.

## 🚀 Projects Included

### 🔰 Beginner Level

#### 1. ResNet50 Image Classifier with Flask 🖼️
**Project Title:** Image tagging using tensorflow model

A simple web application that uses TensorFlow and the ResNet50 pre-trained model to classify images into various categories from the ImageNet dataset.

**✨ Features:**
- Modern, colorful responsive user interface
- Upload any image and get top 5 classification results
- Shows confidence scores for each prediction
- Option to download predictions as a CSV report
- Uses ResNet50, a powerful deep learning model for image classification
- Built with Flask for backend and HTML/CSS for frontend

**🛠️ Technologies Used:**
- Python 3.8+
- Flask (Backend framework)
- TensorFlow / Keras (Machine learning model)
- ResNet50 (Pre-trained on ImageNet)
- Pillow (Image processing)
- NumPy (Array operations)
- HTML5, CSS3 (Frontend)

**📂 Project Structure:**
```
resnet50-image-classifier/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Frontend HTML template
├── static/
│   └── style.css         # CSS styling
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

**⚙️ How It Works:**
1. The Flask backend loads the ResNet50 model pre-trained on ImageNet
2. User uploads an image through the interface
3. The image is resized to 224×224 pixels and preprocessed
4. ResNet50 processes the image and returns the top 5 predicted classes with confidence scores
5. Results are displayed along with the uploaded image
6. Users can download the predictions as a CSV file

**📄 Model Information:**
This application uses ResNet50, a deep convolutional neural network with 50 layers, trained on the ImageNet dataset containing 1,000 object categories. It can classify a wide range of images including animals, vehicles, tools, and natural scenes.

**⚠️ Limitations:**
- Classification accuracy depends on image quality
- Best results with clear, centered subjects
- Some uncommon or abstract objects may not be recognized correctly

**🔮 Future Improvements:**
- Add support for real-time webcam capture
- Implement custom trained models for specific domains
- Store and display classification history for users
- Support batch classification of multiple images
- Enhance UI with drag & drop image uploads

---

### 🔶 Intermediate Level

#### 2. Car Price Prediction System 🚗
**Project Title:** Car Price Prediction System

A comprehensive Machine Learning web application for predicting car selling prices based on various features including fuel type, year of manufacture, kilometers driven, transmission type, and more.

**✨ Features:**
- **Price Prediction:** Get accurate selling price predictions for cars
- **Data Analysis:** Explore dataset statistics and distributions
- **Model Information:** View feature importance and model performance
- **Modern UI:** Beautiful, responsive web interface
- **Real-time Predictions:** Instant price predictions using trained ML model

**📊 Features Used for Prediction:**
- **Year of Manufacture:** Manufacturing year of the car
- **Present Price:** Current market price of the car (in Lakhs)
- **Kilometers Driven:** Total distance covered by the car
- **Fuel Type:** Petrol, Diesel, or CNG
- **Seller Type:** Dealer or Individual
- **Transmission:** Manual or Automatic
- **Number of Previous Owners:** 0, 1, 2, or 3

**🛠️ Technology Stack:**
- **Backend:** Python Flask
- **Machine Learning:** Scikit-learn (Random Forest)
- **Frontend:** HTML, CSS, JavaScript, Bootstrap 5
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib

**📂 Project Structure:**
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

**📈 Model Performance:**
The Random Forest model typically achieves:
- **R² Score:** ~0.85-0.90
- **RMSE:** ~1.5-2.0 Lakhs
- **MAE:** ~1.0-1.5 Lakhs

**🔗 API Endpoints:**
- `GET /`: Main web interface
- `POST /predict`: Get price prediction
- `GET /api/data_analysis`: Get dataset statistics
- `GET /api/model_info`: Get model information
- `POST /train`: Retrain the model
- `GET /load_model`: Load existing model

**🔮 Future Enhancements:**
- Add more ML algorithms
- Implement model comparison
- Add data visualization charts
- Support for batch predictions
- Export predictions to CSV
- User authentication system
- Prediction history tracking

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation & Setup

#### For ResNet50 Image Classifier (Beginner Level)

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/shadowfox-ai-ml-projects.git
cd shadowfox-ai-ml-projects/resnet50-image-classifier
```

2️⃣ **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Run the application**
```bash
python app.py
```

5️⃣ **Open in browser**
```
http://127.0.0.1:5000
```

#### For Car Price Prediction System (Intermediate Level)

1️⃣ **Clone or Download the Project**
```bash
# If using git
git clone <repository-url>
cd car_price_prediction

# Or download and extract the project files
```

2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Download the Dataset**
The application will automatically attempt to download the dataset from Google Drive. If it fails, you can manually download it:

- Visit the dataset link: https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view?usp=drive_link
- Download the CSV file
- Place it in the `data/` folder as `car_data.csv`

4️⃣ **Train the Model**
```bash
python model.py
```
This will:
- Load and preprocess the dataset
- Train a Random Forest model
- Save the trained model to `models/car_price_model.pkl`

5️⃣ **Run the Application**
```bash
python app.py
```
The web application will be available at: `http://localhost:5000`

---

## 📖 Usage

### ResNet50 Image Classifier
1. Open the web application in your browser
2. Click "Choose File" to upload an image
3. Click "Classify Image" to get predictions
4. View the top 5 classification results with confidence scores
5. Optionally download the results as a CSV file

### Car Price Prediction System
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
4. Navigate to "Data Analysis" tab to view dataset statistics
5. Check "Model Info" tab for feature importance and performance metrics

---

## 🛠️ Customization

### Adding New Features to Car Price Prediction
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

---

## 🔧 Troubleshooting

### Common Issues

**Dataset not found:**
- Ensure `car_data.csv` is in the `data/` folder

**Model not trained:**
- Run `python model.py` to train the model

**Port already in use:**
- Change the port in `app.py` or kill the existing process


## 👨‍💻 Developed by

**Chandana C M** © 2025



*This collection is part of the ShadowFox internship program, showcasing practical applications of AI and Machine Learning technologies.*

