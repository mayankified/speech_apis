from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.audio_processing import convert_to_wav
# from app. utils.audio
from app.preprocessing import emotion_preprocessing, lung_preprocessing
from app.models import emotion_model, lung_model

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "emotion"):
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/vnd.wave"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV file.")

    
    try:
        contents = await file.read()
        # Convert the uploaded file to WAV format if needed
        wav_file = convert_to_wav(contents)
        print(wav_file)
        # Preprocess the audio and predict based on model_type
        if model_type == "emotion":
            features = emotion_preprocessing.extract_features(wav_file)
            prediction = emotion_model.predict(features)
        elif model_type == "lung":
            features = lung_preprocessing.extract_features(wav_file)
            prediction = lung_model.predict(features)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type specified.")
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
