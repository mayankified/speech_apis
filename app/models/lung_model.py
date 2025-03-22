import tensorflow as tf
import numpy as np
import pickle
import os

# Global variables for the model and scaler
model = None
scaler = None

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "best_modellung.h5")
scaler_path = os.path.join(current_dir, "scalerlung.pkl")

def load_model():
    global model, scaler
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        print("Lung model loaded successfully!")
        
        # Load the saved scaler using the constructed path
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        # Recompile the model for inference (if necessary)
        model.compile(optimizer="adam", 
                      loss="sparse_categorical_crossentropy", 
                      metrics=["sparse_categorical_accuracy"])
    except Exception as e:
        print("Error loading lung model or scaler:", e)

def predict(features):
    global model, scaler
    if model is None or scaler is None:
        raise ValueError("Lung model or scaler is not loaded.")
    
    # Ensure the features have the correct shape (batch_size, num_features)
    features = np.array(features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    # print(features_scaled)
    # Run prediction using the loaded model
    predictions = model.predict(features_scaled)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Map the predicted class to a human-readable label
    label_mapping = {0: "🟢 Normal", 1: "🔴 Positive"}
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    
    # Return the predicted label and the raw prediction values
    return {"predicted_label": predicted_label, "raw_prediction": predictions.tolist()}
