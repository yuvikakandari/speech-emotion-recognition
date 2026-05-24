import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder

# LOAD DATASET CSV
df = pd.read_csv("dataset.csv")

# LISTS TO STORE FEATURES AND LABELS
X_mfcc = []
X_mel = []
y = []

# AUDIO SETTINGS
SAMPLE_RATE = 22050

DURATION = 3

SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# PROCESS EACH AUDIO FILE
for index, row in df.iterrows():

    file_path = row["path"]

    emotion = row["emotion"]

    try:

        # LOAD AUDIO
        signal, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        # FIX AUDIO LENGTH
        if len(signal) > SAMPLES_PER_TRACK:

            signal = signal[:SAMPLES_PER_TRACK]

        else:

            padding = SAMPLES_PER_TRACK - len(signal)

            signal = np.pad(
                signal,
                (0, padding),
                mode='constant'
            )

        # NORMALIZE AUDIO
        signal = librosa.util.normalize(signal)

        # FEATURE EXTRACTION FOR EACH VERSION
        for augmented_signal in augmented_signals:

            # MFCC EXTRACTION

            mfcc = librosa.feature.mfcc(
                y=augmented_signal,
                sr=sr,
                n_mfcc=40
            )

             # MEL SPECTROGRAM EXTRACTION
            mel = librosa.feature.melspectrogram(
                y=augmented_signal,
                sr=sr
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # STORE FEATURES
            X_mfcc.append(mfcc)

            X_mel.append(mel_db)

            y.append(emotion)

        print(f"Processed: {file_path}")

    except Exception as e:

        print(f"Error processing {file_path}: {e}")

# CONVERT TO NUMPY ARRAYS
X_mfcc = np.array(X_mfcc)

X_mel = np.array(X_mel)

y = np.array(y)

# ENCODE LABELS
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# DISPLAY FINAL SHAPES
print("\nFeature Extraction Completed")

print("\nMFCC Shape:")
print(X_mfcc.shape)

print("\nMel Spectrogram Shape:")
print(X_mel.shape)

print("\nLabels Shape:")
print(y_encoded.shape)

print("\nEmotion Classes:")
print(encoder.classes_)
