import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. INITIAL SETUP, AUDIO CONFIGURATIONS, & AUGMENTATION FUNCTIONS
# =====================================================================

# Load the CSV metadata file containing columns for audio file paths and their emotion labels
df = pd.read_csv("dataset.csv")

# Initialize empty lists to hold the extracted acoustic matrices for each audio file
X_mfcc = []    # Will store 2D matrices representing MFCCs over time
X_mel = []     # Will store 2D matrices representing Mel Spectrogram (in decibels) over time
X_chroma = []  # Will store 2D matrices representing Chroma energy distributions over time
y = []         # Will store the raw text strings of emotions (e.g., 'happy', 'sad')

# Define the standard audio sampling configuration
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ---------------------------------------------------------------------
# DEFINE DATA AUGMENTATION TECHNIQUES (Applied directly to the 1D raw audio signal)
# ---------------------------------------------------------------------

def add_white_noise(signal, noise_factor=0.005):
    """
    Injects random Gaussian white noise into the audio track.
    Simulates real-world radio static and field communications for tactical robustness.
    """
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    return augmented_signal

def pitch_shift(signal, sr, n_steps=2):
    """
    Shifts the pitch up or down without altering the speed of the speech.
    Simulates variance in vocal tract sizes, helping the network generalize across genders/ages.
    n_steps=2 shifts the audio up by 2 semitones.
    """
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

def time_stretch(signal, rate=1.2):
    """
    Speeds up or slows down the talking speed without altering pitch.
    Simulates talking speed changes caused by anxiety, panic, or depression.
    rate=1.2 speeds up the utterance by 20%.
    """
    return librosa.effects.time_stretch(y=signal, rate=rate)


# ---------------------------------------------------------------------
# DEFINE FEATURE EXTRACTION HELPER FUNCTION
# ---------------------------------------------------------------------
def extract_features_from_signal(signal, sr):
    """
    Helper function that takes a standardized 1D signal and extracts MFCC, Mel, and Chroma.
    Ensures absolute consistency across both original and augmented signals.
    """
    # 1. MFCC EXTRACTION (Timbre Features)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

    # 2. MEL SPECTROGRAM EXTRACTION (Frequency Textures)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 3. CHROMA FEATURE EXTRACTION (Pitch/Harmonic Profiling)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12)
    
    return mfcc, mel_db, chroma


# =====================================================================
# 2. ITERATIVE FEATURE EXTRACTION LOOP WITH AUGMENTATION PIPELINE
# =====================================================================

# Loop through every single row of the dataframe to locate, load, format, and extract features from audio files
for index, row in df.iterrows():

    file_path = row["path"]
    emotion = row["emotion"]

    try:
        # STEP A: LOAD AUDIO
        # Load the audio file into memory as a 1D float array (signal)
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # -------------------------------------------------------------
        # GENERATE A DICTIONARY OF TRACK VARIATIONS (The Augmentation Core)
        # -------------------------------------------------------------
        # We apply the transformation to the raw signal FIRST.
        # This gives us 4 distinct versions of the raw audio waveform.
        variations = {
            "original": signal,
            "noisy": add_white_noise(signal, noise_factor=0.004),
            "pitched": pitch_shift(signal, sr=sr, n_steps=2),
            "stretched": time_stretch(signal, rate=1.15)
        }

        # Loop through each variation, standardize its length, and extract features
        for var_name, var_signal in variations.items():
            
            # STEP B: LENGTH ADJUSTMENT (PADDING/CLIPPING)
            # Crucial: Time-stretching changes the length of the signal array,
            # so we must fix the sample length inside this loop for EVERY variation.
            if len(var_signal) > SAMPLES_PER_TRACK:
                var_signal = var_signal[:SAMPLES_PER_TRACK]
            else:
                padding = SAMPLES_PER_TRACK - len(var_signal)
                var_signal = np.pad(var_signal, (0, padding), mode='constant')

            # STEP C: AMPLITUDE NORMALIZATION
            var_signal = librosa.util.normalize(var_signal)

            # STEP D, E, F: RUN FEATURE EXTRACTION
            mfcc, mel_db, chroma = extract_features_from_signal(var_signal, sr)

            # STEP G: CACHE FEATURES AND TARGET LABEL
            X_mfcc.append(mfcc)
            X_mel.append(mel_db)
            X_chroma.append(chroma)
            y.append(emotion)

        print(f"Processed Original + Augmented Variations for: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# =====================================================================
# 3. ARRAY CONVERSION & LABEL ENCODING
# =====================================================================

# Convert Python lists containing 2D arrays into fully vectorized, contiguous 3D NumPy arrays
X_mfcc = np.array(X_mfcc)
X_mel = np.array(X_mel)
X_chroma = np.array(X_chroma)
y = np.array(y)

# Instantiate a Scikit-Learn LabelEncoder to transform categorical text strings into numeric class IDs
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# =====================================================================
# 4. DIAGNOSTIC DATA VERIFICATION
# =====================================================================

print("\n" + "="*50)
print("AUGMENTED FEATURE EXTRACTION PIPELINE COMPLETED")
print("="*50)

print(f"MFCC Shape:             {X_mfcc.shape}   -> Expected: [Samples, 40, Time_Frames]")
print(f"Mel Spectrogram Shape:  {X_mel.shape}    -> Expected: [Samples, 128, Time_Frames]")
print(f"Chroma STFT Shape:      {X_chroma.shape} -> Expected: [Samples, 12, Time_Frames]")
print(f"Encoded Target Shape:   {y_encoded.shape}   -> Expected: [Samples,]")
print(f"Identified Emotion Classes: {encoder.classes_}")


# =====================================================================
# 5. SERIALIZE AND SAVE COMPILED DATASET
# =====================================================================

np.save("X_mfcc.npy", X_mfcc)
np.save("X_mel.npy", X_mel)
np.save("X_chroma.npy", X_chroma)
np.save("y.npy", y_encoded)

print("\nAugmented arrays successfully serialized and saved to disk.")