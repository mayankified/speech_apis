import os
import tensorflow as tf
import numpy as np
import pickle

# Label mapping for emotions
label_mapping = {
    "happyness": 0,
    "neutral": 1,
    "anger": 2,
    "sadness": 3,
    "fear": 4,
    "boredom": 5,
    "disgust": 6,
}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Global model variable
model = None
scaler = None
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model_modb.keras")
scaler_path = os.path.join(current_dir, "scaler_emodb.pkl")

def load_model():
    """
    Defines and loads the emotion model.
    The model architecture is defined and then pre-trained weights are loaded.
    """
    global model, scaler  # Add scaler here!
    try:
        # Load the model directly from the file
        model = tf.keras.models.load_model(model_path)
        print("Emotion model loaded successfully!")
        # Load the saved scaler using the constructed path
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("Emo Scaler loaded successfully!")
        optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimiser,
            loss="sparse_categorical_crossentropy",
            metrics=["SparseCategoricalAccuracy"],
        )
        print("Emotion model weights loaded successfully!")
    except Exception as e:
        print("Error loading emotion model weights:", e)

def predict(features):
    """
    Given a feature vector, runs the model prediction and returns the predicted emotion.
    
    Parameters:
        features: A list of 90 features extracted from the audio.
    
    Returns:
        A dictionary with the predicted emotion.
    """
    global model, scaler
    if model is None:
        raise ValueError("Emotion model is not loaded.")
    
    # Convert the feature list to a NumPy array with correct type and shape.
    features = np.array(features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    # print(features_scaled)
    # Run prediction using the loaded model
    predictions = model.predict(features_scaled)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_emotion = reverse_label_mapping.get(predicted_index, "Unknown")
    return {"predicted_emotion": predicted_emotion}
