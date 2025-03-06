EcoSort AI 🌍♻️
AI-Powered Waste Classification System
EcoSort AI is an AI-driven waste classification system that helps users identify whether a waste item is Recyclable or Organic using image recognition. 
Simply upload a photo, and the system provides real-time classification to promote better waste management and a cleaner environment.

🚀 Features
✅ Real-time Waste Classification – Upload an image and get instant results
✅ Confidence Score – Displays how confident the AI is in its prediction
✅ Simple & Responsive UI – Built with HTML, CSS, and JavaScript
✅ AI-Powered Backend – Uses a trained CNN model with Flask for processing

📌 Tech Stack
Frontend: HTML, CSS, JavaScript
Backend: Python (Flask)
AI Model: TensorFlow/Keras CNN Model


📂 Project Directory Structure
waste_classification_website/

├── app.py                   # Flask backend for handling requests

├── model.py                 # Waste Classification model building

├── requirements.txt         # Dependencies for the project

├── Waste_Model.h5           # Pre-trained AI model for waste classification

├── templates/               # HTML templates for frontend

   ├── index.html           # Homepage with image upload feature

   └── result.html          # Displays classification results

└── static/                  # Static files (CSS, uploads)
    ├── styles.css           # Stylesheet for UI
    └── uploads/             # Directory for storing uploaded images


    
📸 How It Works
Upload an image of a waste item on the homepage.
The AI model processes the image and classifies it as Recyclable or Organic.
The result page displays the classification along with a confidence score.

