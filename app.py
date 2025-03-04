from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import logging
from PIL import Image
import io

# Load the trained model
model = load_model('Waste_Model.h5')

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define prediction function directly to avoid OpenCV issues
def predict_func(img_array):
    """
    Make prediction using the model without OpenCV dependencies
    
    Args:
        img_array: Normalized numpy array of shape (1, 224, 224, 3)
        
    Returns:
        prediction: Class label
        confidence: Prediction confidence
    """
    # Ensure img_array is properly formatted
    if img_array.shape != (1, 224, 224, 3):
        raise ValueError(f"Expected input shape (1, 224, 224, 3), got {img_array.shape}")
        
    # Get predictions
    predictions = model.predict(img_array)
    
    # Get class label and confidence
    class_names = ['organic', 'recyclable']  # Update with your actual classes
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(predictions[0][predicted_class] * 100)
    
    return class_names[predicted_class], confidence

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
   
    file = request.files['file']
   
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
    if file:
        try:
            # Read file into memory
            file_bytes = file.read()
            if not file_bytes:
                logging.error("❌ Empty file uploaded")
                return jsonify({'error': 'Empty file uploaded'}), 400
                
            # Process image using PIL exclusively (no OpenCV)
            img = Image.open(io.BytesIO(file_bytes))
            img = img.convert('RGB')  # Ensure image is in RGB format
            
            # Resize using PIL
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Reshape for model input
            img_array = np.reshape(img_array, (1, 224, 224, 3))
            
            # Save the image to a static folder
            image_path = f'static/uploads/{file.filename}'
            img.save(image_path)
            image_url = url_for('static', filename=f'uploads/{file.filename}')
            
            # Make prediction
            try:
                prediction, confidence = predict_func(img_array)
                logging.info(f"✅ Prediction successful: {prediction} with {confidence:.2f}% confidence")
                return render_template('result.html', image_url=image_url, prediction=prediction, confidence=confidence)
            except Exception as e:
                logging.error(f"❌ Prediction error: {str(e)}")
                return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
                
        except Exception as e:
            logging.error(f"❌ Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
