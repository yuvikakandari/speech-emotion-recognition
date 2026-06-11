import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Import custom attention layer so Keras knows how to read it
from train_model import AttentionLayer 

print("Loading test data for evaluation...")
X_mfcc = np.load("X_mfcc.npy")
X_delta = np.load("X_delta.npy")
X_deltadelta = np.load("X_deltadelta.npy")
X_mel = np.load("X_mel.npy")
y_raw = np.load("y_raw_text.npy")

# Reconstruct identical input shapes
X_combined = np.concatenate((X_mfcc, X_delta, X_deltadelta, X_mel), axis=1)
X_final = np.expand_dims(X_combined, axis=-1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)

# Split exactly the same way as training (using random_state=42)
_, X_test, _, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Loading the optimized model configuration asset...")
# Pass custom_objects so Keras registers your custom Attention mechanism
model = tf.keras.models.load_model(
    "best_ser_hybrid_model.keras", 
    custom_objects={'AttentionLayer': AttentionLayer}
)

print("Executing test set inference...")
y_pred_probabilities = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probabilities, axis=1)

# 1. GENERATE THE CLASSIFICATION REPORT (Precision, Recall, F1)
print("\n" + "="*50)
print("            PHASE 1 EVALUATION REPORT")
print("="*50)
emotion_names = encoder.classes_
print(classification_report(y_test, y_pred_labels, target_names=emotion_names))


# 2. GENERATE AND SAVE THE CONFUSION MATRIX GRAPH
cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=emotion_names, 
    yticklabels=emotion_names
)
plt.title("Acoustic Emotion Recognition - Confusion Matrix", fontsize=14, pad=15)
plt.xlabel("Predicted Emotion Labels", fontsize=12, labelpad=10)
plt.ylabel("Actual True Labels", fontsize=12, labelpad=10)
plt.tight_layout()

# Save the plot directly as an image 
plt.savefig("ser_confusion_matrix.png", dpi=300)
print("\nSuccess! Confusion Matrix visualization exported as 'ser_confusion_matrix.png'")
plt.show()