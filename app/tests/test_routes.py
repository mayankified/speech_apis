import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_invalid_file():
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", b"Not an audio file", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_valid_emotion_file():
    # Create a dummy WAV file for testing
    wav_bytes = create_dummy_wav()
    response = client.post(
        "/predict", 
        data={"model_type": "emotion"}, 
        files={"file": ("test.wav", wav_bytes, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

def create_dummy_wav():
    import io, wave
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b'\x00\x00' * 44100)
    byte_io.seek(0)
    return byte_io.read()
