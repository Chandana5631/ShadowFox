# Project Title
Image tagging using tensorflow model
🖼️ ResNet50 Image Classifier with Flask

A simple web application that uses TensorFlow and the ResNet50 pre-trained model to classify 
images into various categories from the ImageNet dataset.

✨ Features

Modern, colorful responsive user interface

Upload any image and get top 5 classification results

Shows confidence scores for each prediction

Option to download predictions as a CSV report

Uses ResNet50, a powerful deep learning model for image classification

Built with Flask for backend and HTML/CSS for frontend

🛠️ Technologies Used

Python 3.8+

Flask (Backend framework)

TensorFlow / Keras (Machine learning model)

ResNet50 (Pre-trained on ImageNet)

Pillow (Image processing)

NumPy (Array operations)

HTML5, CSS3 (Frontend)

🚀 Getting Started
Prerequisites

Python 3.8 or higher

pip (Python package installer)

Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/yourusername/resnet50-image-classifier.git
cd resnet50-image-classifier


2️⃣ Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Run the application

python app.py


5️⃣ Open in browser

http://127.0.0.1:5000

⚙️ How It Works

The Flask backend loads the ResNet50 model pre-trained on ImageNet.

User uploads an image through the interface.

The image is resized to 224×224 pixels and preprocessed.

ResNet50 processes the image and returns the top 5 predicted classes with confidence scores.

Results are displayed along with the uploaded image.

Users can download the predictions as a CSV file.

📄 Model Information

This application uses ResNet50, a deep convolutional neural network with 50 layers, trained on 
the ImageNet dataset containing 1,000 object categories.
It can classify a wide range of images including animals, vehicles, tools, and natural scenes.

📂 Project Structure
├── app.py                # Flask backend
├── templates/
│   └── index.html         # Frontend HTML template
├── static/
│   └── style.css          # CSS styling
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

⚠️ Limitations

Classification accuracy depends on image quality.

Best results with clear, centered subjects.

Some uncommon or abstract objects may not be recognized correctly.

🔮 Future Improvements

Add support for real-time webcam capture.

Implement custom trained models for specific domains.

Store and display classification history for users.

Support batch classification of multiple images.

Enhance UI with drag & drop image uploads.

👨‍💻 Developed by

Chandana C M © 2025

