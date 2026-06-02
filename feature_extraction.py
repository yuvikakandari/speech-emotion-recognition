import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. INITIAL SETUP AND AUDIO CONFIGURATIONS
# ==========================================

# Load the CSV metadata file containing columns for audio file paths and their emotion labels
df = pd.read_csv("dataset.csv")

# Initialize empty lists to hold the extracted acoustic matrices for each audio file
X_mfcc = []    # Will store 2D matrices representing MFCCs over time
X_mel = []     # Will store 2D matrices representing Mel Spectrogram (in decibels) over time
X_chroma = []  # NEW: Will store 2D matrices representing Chroma energy distributions over time
y = []         # Will store the raw text strings of emotions (e.g., 'happy', 'sad')

# Define the standard audio sampling configuration
# 22050 Hz is standard for speech processing; it captures up to an 11,025 Hz frequency range (Nyquist theorem)
SAMPLE_RATE = 22050

# Target length for each audio segment in seconds
DURATION = 3

# Compute total expected data points (samples) per audio clip to ensure uniform shape across the entire dataset
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


# ==========================================
# 2. ITERATIVE FEATURE EXTRACTION LOOP
# ==========================================

# Loop through every single row of the dataframe to locate, load, format, and extract features from audio files
for index, row in df.iterrows():

    file_path = row["path"]
    emotion = row["emotion"]

    try:
        # ----------------------------------
        # STEP A: LOAD AND STANDARDIZE AUDIO
        # ----------------------------------
        
        # Load the audio file into memory as a 1D float array (signal)
        # Setting sr=SAMPLE_RATE forces librosa to resample any audio clip to 22,050 Hz automatically
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # ----------------------------------
        # STEP B: LENGTH ADJUSTMENT (PADDING/CLIPPING)
        # ----------------------------------
        # Deep learning architectures (like CNNs and LSTMs) demand perfectly uniform input matrices.
        # We must explicitly force every audio signal to have exactly SAMPLES_PER_TRACK length.
        
        if len(signal) > SAMPLES_PER_TRACK:
            # Scenario 1: Audio is too long. Slice the array from index 0 up to SAMPLES_PER_TRACK
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            # Scenario 2: Audio is too short. Calculate how many data points are missing
            padding = SAMPLES_PER_TRACK - len(signal)
            # Pad the right side of the 1D array with constant zero values (silence padding)
            signal = np.pad(signal, (0, padding), mode='constant')

        # ----------------------------------
        # STEP C: AMPLITUDE NORMALIZATION
        # ----------------------------------
        # Rescales peak audio values to fall precisely between -1.0 and 1.0. 
        # This prevents variance in recording volume/microphone sensitivity from biasing our network.
        signal = librosa.util.normalize(signal)


        # ----------------------------------
        # STEP D: MFCC EXTRACTION (Timbre Features)
        # ----------------------------------
        # Extracts Mel-Frequency Cepstral Coefficients, which capture vocal tract shapes/timbre.
        # Outputs a 2D matrix: [n_mfcc x time_frames] (Default frame size is 2048 samples with 512 hop length)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40  # 40 coefficients are standard for capturing highly descriptive human speech dynamics
        )


        # ----------------------------------
        # STEP E: MEL SPECTROGRAM EXTRACTION (Frequency Textures)
        # ----------------------------------
        # Step 1: Compute power map distributed across the non-linear human auditory Mel-scale
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr
        )
        # Step 2: Convert linear power values to logarithmic Decibel (dB) units. 
        # This closely matches how human ears experience differences in loudness/volume.
        mel_db = librosa.power_to_db(
            mel,
            ref=np.max  # Scales the highest volume peak to 0 dB; all other data steps become negative values
        )


        # ----------------------------------
        # STEP F: CHROMA FEATURE EXTRACTION (Pitch/Harmonic Profiling)
        # ----------------------------------
        # NEW FEATURE: Extracts a "Chroma Vector" (Chroma Short-Time Fourier Transform).
        # It projects the full spectrum of frequencies down into 12 discrete bins representing 
        # the 12 semitones/musical pitches of an octave (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
        # In speech, chroma helps identify structural shifts in emotional intonation, melody, and micro-tonality.
        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=sr,
            n_chroma=12  # Explicitly fixes output rows to the 12 semantic pitch categories
        )


        # ----------------------------------
        # STEP G: CACHE FEATURES AND LABELS
        # ----------------------------------
        # Append the calculated 2D feature matrices into their respective storage lists
        X_mfcc.append(mfcc)
        X_mel.append(mel_db)
        X_chroma.append(chroma)  # Cache the newly extracted chroma matrix
        
        # Keep track of the matching textual target label
        y.append(emotion)

        print(f"Processed: {file_path}")

    except Exception as e:
        # Prevents one corrupted audio file from crashing an hours-long preprocessing pipeline
        print(f"Error processing {file_path}: {e}")


# ==========================================
# 3. ARRAY CONVERSION & LABEL ENCODING
# ==========================================

# Convert Python lists containing 2D arrays into fully vectorized, contiguous 3D NumPy arrays
# Final Shape Pattern: [Number_of_Samples, Rows/Features, Time_Steps]
X_mfcc = np.array(X_mfcc)
X_mel = np.array(X_mel)
X_chroma = np.array(X_chroma)  # Vectorize chroma arrays to a matching 3D block
y = np.array(y)

# Instantiate a Scikit-Learn LabelEncoder to transform categorical text strings into numeric class IDs
# E.g., ['Angry', 'Happy', 'Sad'] becomes integers [0, 1, 2] so loss functions can process them natively
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# ==========================================
# 4. DIAGNOSTIC DATA VERIFICATION
# ==========================================

print("\n" + "="*40)
print("FEATURE EXTRACTION PIPELINE COMPLETED")
print("="*40)

# Print structural shapes to ensure formatting is entirely uniform and error-free
print(f"MFCC Shape:             {X_mfcc.shape}   -> Expected: [Samples, 40, Time_Frames]")
print(f"Mel Spectrogram Shape:  {X_mel.shape}    -> Expected: [Samples, 128, Time_Frames]")
print(f"Chroma STFT Shape:      {X_chroma.shape} -> Expected: [Samples, 12, Time_Frames]")
print(f"Encoded Target Shape:   {y_encoded.shape}   -> Expected: [Samples,]")
print(f"Identified Emotion Classes: {encoder.classes_}")


# ==========================================
# 5. SERIALIZE AND SAVE COMPILED DATASET
# ==========================================

# Save the structured arrays directly to your local workspace as fast, binary `.npy` storage files.
# This ensures you can load them instantly inside your training notebooks without re-running librosa features.
np.save("X_mfcc.npy", X_mfcc)
np.save("X_mel.npy", X_mel)
np.save("X_chroma.npy", X_chroma)  # Save the new compiled chroma dataset matrix
np.save("y.npy", y_encoded)

print("\nAll arrays successfully serialized and saved to disk.")