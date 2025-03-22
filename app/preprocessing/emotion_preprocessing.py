import numpy as np
import librosa
from scipy import signal as spsignal
from vmdpy import VMD

def energy(audio, frame_length, hop_length):
    energies = []
    for i in range(0, len(audio), hop_length):
        frame = audio[i:i+frame_length]
        rms_current_frame_energy = np.sum(frame ** 2)
        energies.append(rms_current_frame_energy)
    return np.mean(np.array(energies))

def rms(audio, frame_length, hop_length):
    # Computes the RMS for the last frame (if needed)
    for i in range(0, len(audio), hop_length):
        rms_energy_current_frame = np.sqrt(np.sum(audio[i:i+frame_length] ** 2) / frame_length)
    return rms_energy_current_frame

def extract_features(wav_file):
    """
    Extracts 90 features from a given WAV file using VMD and MFCCs.
    
    Parameters:
        wav_file: A file-like object or a path to the WAV file.
        
    Returns:
        A list of 90 features extracted from the audio.
    """
    # Load the audio signal; librosa can handle file paths or file-like objects
    audio, sr = librosa.load(wav_file, sr=16000)
    
    # Apply pre-emphasis
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    
    # Normalize the audio signal
    audio = librosa.util.normalize(audio)
    
    # Define frame parameters and windowing
    hop_length = 400
    frame_length = 800
    window = spsignal.windows.hann(frame_length)
    
    # VMD parameters
    alpha = 5000    # Moderate bandwidth constraint
    tau = 0         # Noise-tolerance (no strict fidelity enforcement)
    K = 3           # Number of modes; 3 modes will yield 3 * 30 = 90 features
    DC = 0          # No DC part imposed
    init = 1        # Initialize omegas uniformly
    tol = 1e-7      # Tolerance
    
    features = []
    
    # Process audio in overlapping frames
    for start in range(0, len(audio), hop_length):
        frame = audio[start:start+frame_length]
        # Pad the frame if it's shorter than expected
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        
        # Apply the Hanning window to the frame
        windowed_frame = frame * window
        
        # Perform Variational Mode Decomposition (VMD)
        try:
            u, u_hat, omega = VMD(windowed_frame, alpha, tau, K, DC, init, tol)
        except Exception:
            # If VMD fails on this frame, skip it
            continue
        
        data = []
        # For each decomposed mode, extract MFCC features and compute their mean
        for mode in u:
            mfccs = librosa.feature.mfcc(y=mode, n_fft=frame_length, hop_length=hop_length, sr=sr, n_mfcc=30)
            mfccs_mean = np.mean(mfccs, axis=1)
            data.extend(mfccs_mean.tolist())
        # Append the 90 features (if K=3 and n_mfcc=30) for the current frame
        features.append(data)
    
    # Average features over all processed frames
    if features:
        features_array = np.array(features)
        feature_avg = np.mean(features_array, axis=0)
        return feature_avg.tolist()
    else:
        # If no frames could be processed, return a zero vector of length 90
        return [0] * 90
