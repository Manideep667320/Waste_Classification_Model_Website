import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import cv2
import logging

# Define the number of classes
numberOfClass = 2  # Assuming two classes: Recyclable and Organic

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Image data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Paths to training and testing datasets
train_path = "C:/AICTE/DATASET/DATASET/TRAIN"
test_path = "C:/AICTE/DATASET/DATASET/TEST"

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

if __name__ == "__main__":
    # Train the model
    hist = model.fit(
        train_generator,
        steps_per_epoch=150,
        epochs=10,
        validation_data=test_generator,
        validation_steps=150,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    # Log training and validation metrics
    logging.info(f"Training accuracy: {hist.history['accuracy'][-1]}")
    logging.info(f"Validation accuracy: {hist.history['val_accuracy'][-1]}")
    logging.info(f"Training loss: {hist.history['loss'][-1]}")
    logging.info(f"Validation loss: {hist.history['val_loss'][-1]}")

    # Save the trained model
    model.save("Waste_Model.keras")

    # Plot training history
    plt.figure(figsize=[10, 6])
    plt.plot(hist.history["accuracy"], label="Train Accuracy")
    plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=[10, 6])
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

# Function for making predictions
def predict_func(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        logging.error("Input image is None. Please check the image path.")
        return "Error: Image not found", 0

    logging.info(f"Input image shape: {img.shape if img is not None else 'None'}")
    logging.info(f"Image loaded from path: {img_path}")

    # Resize and reshape the image
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3)) / 255.0  # Add batch dimension and normalize
    prediction = model.predict(img)
    
    # Log the raw prediction scores
    logging.info(f"Raw predictions: {prediction}")

    result = np.argmax(prediction)
    confidence = np.max(prediction)

    # Log the predicted class and confidence score
    logging.info(f"Predicted class: {result}, Confidence: {confidence}")

    if result == 0:
        return "Recyclable", confidence
    elif result == 1:
        return "Organic", confidence
