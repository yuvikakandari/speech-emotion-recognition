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

