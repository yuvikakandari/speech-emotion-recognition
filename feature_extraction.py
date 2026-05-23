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