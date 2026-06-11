import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# =====================================================================
# 1. COPIED CUSTOM ATTENTION LAYER (NO IMPORTS NEEDED)
# =====================================================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal", 
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.dot(x, self.W)
        et = K.squeeze(et, axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# =====================================================================
# 2. LOAD & RECONSTRUCT IDENTICAL TEST SET SPLIT
# =====================================================================
print("Loading holdout validation matrices...")
X_mfcc = np.load("X_mfcc.npy")
X_delta = np.load("X_delta.npy")
X_deltadelta = np.load("X_deltadelta.npy")
X_mel = np.load("X_mel.npy")
y_raw = np.load("y_raw_text.npy")

X_combined = np.concatenate((X_mfcc, X_delta, X_deltadelta, X_mel), axis=1)
X_final = np.expand_dims(X_combined, axis=-1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)

# Split using identical random state to isolate the true 20% validation split
_, X_test, _, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# =====================================================================
# 3. DIRECT WEIGHTS RESTORATION (FAST INFERENCE ONLY)
# =====================================================================
print("\nLoading saved optimal weights file...")
# Passing the class here cleanly without invoking train_model.py execution loops
model = tf.keras.models.load_model(
    "best_ser_hybrid_model.keras", 
    custom_objects={'AttentionLayer': AttentionLayer}
)

print("Calculating predictions over test matrices (No epochs required!)...")
y_pred_probabilities = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probabilities, axis=1)


# =====================================================================
# 4. PRINT REPORT AND PLOT GRAPH
# =====================================================================
print("\n" + "="*50)
print("             DRDO PHASE 1 EVALUATION REPORT")
print("="*50)
emotion_names = encoder.classes_
print(classification_report(y_test, y_pred_labels, target_names=emotion_names))

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Acoustic Emotion Recognition - Confusion Matrix", fontsize=14, pad=15)
plt.xlabel("Predicted Emotion Labels", fontsize=12, labelpad=10)
plt.ylabel("Actual True Labels", fontsize=12, labelpad=10)
plt.tight_layout()

plt.savefig("ser_confusion_matrix.png", dpi=300)
print("\nSuccess! Confusion Matrix visualization exported as 'ser_confusion_matrix.png'")
plt.show()