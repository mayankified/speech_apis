import numpy as np
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def extract_features(wav_file):
    """
    Extracts 40 features from a given WAV file for lung disease detection.
    The features are computed by processing the audio in overlapping frames,
    applying a Hanning window, computing MFCCs for each frame (with 40 coefficients),
    and then averaging the MFCCs across all frames.

    Parameters:
        wav_file: A file-like object or path to a WAV file.
    
    Returns:
        A list of 40 averaged MFCC features.
    """
    sr = 16000
    # Load the audio signal; librosa can handle file paths or file-like objects.
    audio, _ = librosa.load(wav_file, sr=sr)
    
    # Apply pre-emphasis filter.
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    
    # Normalize the audio signal.
    audio = librosa.util.normalize(audio)
    
    # Define frame parameters.
    hop_length = 400
    frame_length = 800
    # Create a Hanning window.
    q = signal.windows.hann(frame_length)
    
    features = []
    
    # Process the audio in frames.
    for start in range(0, len(audio), hop_length):
        current_frame = audio[start:start+frame_length]
        # Pad the frame if it's shorter than the desired length.
        if len(current_frame) < frame_length:
            current_frame = np.pad(current_frame, (0, frame_length - len(current_frame)), mode='constant')
        # Apply the Hanning window.
        current_frame_windowed = current_frame * q
        
        # Compute MFCCs; using n_mfcc=40 returns 40 coefficients.
        # Transposing so that we average over time frames.
        mfccs = librosa.feature.mfcc(y=current_frame_windowed, n_fft=frame_length, hop_length=hop_length, sr=sr, n_mfcc=40).T
        mfccs_mean = np.mean(mfccs, axis=0)
        features.append(mfccs_mean.tolist())
    
    if features:
        features_array = np.array(features)
        feature_avg = np.mean(features_array, axis=0)
        return feature_avg.tolist()
    else:
        # If no features were extracted, return a zero vector.
        return [0] * 40
