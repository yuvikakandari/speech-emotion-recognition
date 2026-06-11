import os
import glob
import numpy as np
import librosa

# 1. INITIAL DIRECTORY SETUP & EMOTION DICTIONARY CONFIGURATIONS


BASE_DATASET_DIR = "datasets"
RAVDESS_DIR = os.path.join(BASE_DATASET_DIR, "RAVDESS")
CREMA_DIR = os.path.join(BASE_DATASET_DIR, "CREMA-D")
TESS_DIR = os.path.join(BASE_DATASET_DIR, "TESS")

# Master feature storage lists
X_mfcc = []
X_delta = []
X_deltadelta = []
X_mel = []
y = []

# Audio configuration standards
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# STRATEGIC FIX: Clean 6-Class dictionary (Dropped 02=calm and 08=surprised)
RAVDESS_EMOTIONS = {
    "01": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust"
}

# Crema-D natively uses these 6 balanced classes
CREMA_EMOTIONS = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful", 
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}

# 2. DEFINITIONS OF 3 BASIC AUGMENTATION & EXTRACTION HELPERS

def add_white_noise(signal, noise_factor=0.004):
    """Augmentation 1: Static white noise injection"""
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def pitch_shift(signal, sr, n_steps=2):
    """Augmentation 2: Vocal frequency pitch shift"""
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

def time_stretch(signal, rate=1.15):
    """Augmentation 3: Utterance speech speed stretching"""
    return librosa.effects.time_stretch(y=signal, rate=rate)

def process_and_cache(signal, sr, emotion_label):
    """
    Standardizes signal length, normalizes amplitude, extracts Mel Spectrogram,
    MFCC, Delta MFCC, and Delta-Delta MFCC, then caches them.
    """
    # Fix track duration length uniformly
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(signal)
        signal = np.pad(signal, (0, padding), mode='constant')

    # Normalize volume amplitude peaks
    signal = librosa.util.normalize(signal)

    # 1. Base MFCC Extraction (40 Coefficients)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    # 2. NEW: Delta MFCC (Tracks velocity/first-derivative of spectral changes over time)
    delta_mfcc = librosa.feature.delta(mfcc)

    # 3. NEW: Delta-Delta MFCC (Tracks acceleration/second-derivative of spectral changes)
    deltadelta_mfcc = librosa.feature.delta(mfcc, order=2)

    # 4. Mel Spectrogram Extraction
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Cache calculations into our lists
    X_mfcc.append(mfcc)
    X_delta.append(delta_mfcc)
    X_deltadelta.append(deltadelta_mfcc)
    X_mel.append(mel_db)
    y.append(emotion_label)

# 3. MULTI-DATASET PROCESSING PARSERS

#  PROCESS RAVDESS 
print("Extracting features from RAVDESS (Filtering for 6 classes)...")
ravdess_files = glob.glob(os.path.join(RAVDESS_DIR, "Actor_*", "*.wav"))
for file_path in ravdess_files:
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    if parts[0] == "03" and parts[1] == "01": # Audio-only speech check
        emotion = RAVDESS_EMOTIONS.get(parts[2]) # Skips calm/surprised automatically
        if emotion:
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                process_and_cache(signal, sr, emotion)
                process_and_cache(add_white_noise(signal), sr, emotion)
                process_and_cache(pitch_shift(signal, sr), sr, emotion)
                process_and_cache(time_stretch(signal), sr, emotion)
            except Exception as e: pass

#  PROCESS CREMA-D 
print("Extracting features from CREMA-D (All 6 native classes)...")
crema_files = glob.glob(os.path.join(CREMA_DIR, "*.wav"))
for file_path in crema_files:
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) >= 3:
        emotion = CREMA_EMOTIONS.get(parts[2])
        if emotion:
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                process_and_cache(signal, sr, emotion)
                process_and_cache(add_white_noise(signal), sr, emotion)
                process_and_cache(pitch_shift(signal, sr), sr, emotion)
                process_and_cache(time_stretch(signal), sr, emotion)
            except Exception as e: pass

#  PROCESS TESS 
print("Extracting features from TESS (Filtering out pleasant surprise)...")
tess_folders = glob.glob(os.path.join(TESS_DIR, "*"))
for folder_path in tess_folders:
    if os.path.isdir(folder_path):
        folder_name = os.path.basename(folder_path).lower()
        
        # Explicit mapping to drop surprise folders completely
        if "angry" in folder_name: emotion = "angry"
        elif "disgust" in folder_name: emotion = "disgust"
        elif "fear" in folder_name: emotion = "fearful"
        elif "happy" in folder_name: emotion = "happy"
        elif "neutral" in folder_name: emotion = "neutral"
        elif "sad" in folder_name: emotion = "sad"
        else: emotion = None # Dropping surprise/ps folders
        
        if emotion:
            tess_files = glob.glob(os.path.join(folder_path, "*.wav"))
            for file_path in tess_files:
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    process_and_cache(signal, sr, emotion)
                    process_and_cache(add_white_noise(signal), sr, emotion)
                    process_and_cache(pitch_shift(signal, sr), sr, emotion)
                    process_and_cache(time_stretch(signal), sr, emotion)
                except Exception as e: pass

# 4. VECTORIZATION & DISK SERIALIZATION
X_mfcc = np.array(X_mfcc)
X_delta = np.array(X_delta)
X_deltadelta = np.array(X_deltadelta)
X_mel = np.array(X_mel)
y = np.array(y)

# Save arrays directly to disk
np.save("X_mfcc.npy", X_mfcc)
np.save("X_delta.npy", X_delta)
np.save("X_deltadelta.npy", X_deltadelta)
np.save("X_mel.npy", X_mel)

# Convert labels to text format to re-encode inside train_model.py cleanly
np.save("y_raw_text.npy", y)

print("\n" + "="*50)
print("BALANCED 6-CLASS PIPELINE COMPLETED")
print("="*50)
print(f"Total Fused Samples Generated:   {X_mfcc.shape[0]}")
print(f"MFCC Frame Block Shape:         {X_mfcc.shape}")
print(f"Delta Frame Block Shape:        {X_delta.shape}")
print(f"Delta-Delta Frame Block Shape:  {X_deltadelta.shape}")
print(f"Mel Spectrogram Block Shape:    {X_mel.shape}")


#using 3 datasets now
#3 basic data augmentation techniques
#remove chroma
#use delta mfcc, delta delta mfcc