import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# =====================================================================
# 1. DEFINE THE CUSTOM ATTENTION LAYER ARCHITECTURE
# =====================================================================
class AttentionLayer(Layer):
    """
    Learns which time frames carry the most emotional weight,
    mathematically amplifying expressions while silencing dead noise.
    """
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
# 2. LOAD DATA AND FUSE MULTI-DATASET ACOUSTIC FEATURES
# =====================================================================
print("Loading multi-dataset augmented feature blocks...")

X_mfcc = np.load("X_mfcc.npy")                # Shape: (43592, 40, 130)
X_delta = np.load("X_delta.npy")              # Shape: (43592, 40, 130)
X_deltadelta = np.load("X_deltadelta.npy")    # Shape: (43592, 40, 130)
X_mel = np.load("X_mel.npy")                  # Shape: (43592, 128, 130)
y_raw = np.load("y_raw_text.npy")              # Raw textual tags (43592,)

# Stack all feature tracks vertically along Axis 1 (Rows Axis)
# Math: 40 (MFCC) + 40 (Delta) + 40 (Delta-Delta) + 128 (Mel) = 248 rows
X_combined = np.concatenate((X_mfcc, X_delta, X_deltadelta, X_mel), axis=1) 
# Intermediary Shape: (43592, 248, 130)

# Expand dimensions to add the 4D channel block required by Keras Conv2D layers
X_final = np.expand_dims(X_combined, axis=-1) 
# Final Input Shape: (43592, 248, 130, 1)

print(f"Feature engineering complete. Combined Matrix Shape: {X_final.shape}")

# Encode target text string tags into structural integers (0 through 5)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)


# =====================================================================
# 3. STRATIFIED TRAIN-TEST DATA SPLIT
# =====================================================================
# Stratification guarantees a uniform class distribution between sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)


# =====================================================================
# 4. BLUEPRINT MODEL ASSEMBLY LINE (CNN + BiLSTM + Attention)
# =====================================================================
print("\nAssembling Custom 6-Class Hybrid Neural Network Architecture...")

model = Sequential()

# BLOCK 1: 32 CNN Filters -> Batch Normalization -> Max Pooling
# Injected with 'relu' activation to handle non-linear spatial frequency structures
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(248, 130, 1)))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2)))

# BLOCK 2: 64 CNN Filters -> Batch Normalization -> Max Pooling
# Deep extraction tracking with explicit 'relu' activation boundaries
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2)))

# SEQUENTIAL RESHAPE BRIDGE: Converts 4D image matrices into a 3D timeline tensor
model.add(Reshape(target_shape=(model.output_shape[2], model.output_shape[1] * model.output_shape[3])))

# CHRONOLOGICAL DEPT: Bidirectional LSTM tracking structural velocities -> Dropout
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.4)) 

# CORE FOCUS DEPT: Learned Attention weight alignment across time steps
model.add(AttentionLayer())

# VERDICT CLOSURE DEPT: Feature Mixing -> Dropout Regularization -> Softmax Classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

# Final output targets exactly 6 balanced categories outputting clean probability spreads
model.add(Dense(6, activation='softmax'))


# =====================================================================
# 5. COMPILATION AND LOSS MANAGEMENT
# =====================================================================
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# Render model summary map to trace shape modifications across layers
model.summary()


# =====================================================================
# 6. CONFIGURING INTELLECTUAL AUTOMATION CALLBACK MATRIX
# =====================================================================
# Autosave engine: Backs up top configurations without locking up storage
checkpoint = ModelCheckpoint(
    filepath="best_ser_hybrid_model.keras", 
    monitor="val_loss",
    save_best_only=True,                     
    mode="min",                              
    verbose=1
)

# Adaptive Gearbox: Shrinks learning rate dynamically when steps plateau
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,                              
    patience=3,                              
    min_lr=1e-6,                             
    verbose=1
)

# Emergency Brake: Cuts execution loop short if overfitting pattern is spotted
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,                              
    restore_best_weights=True,               
    verbose=1
)

# Package callbacks into a streamlined runtime execution array
callbacks_list = [checkpoint, reduce_lr, early_stop]


# =====================================================================
# 7. TRAINING LOOP PIPELINE EXECUTION
# =====================================================================
print("\nCommencing big-data training sequence...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,                               # Upper limit threshold boundary
    batch_size=64,                           # High processing throughput configuration
    callbacks=callbacks_list                  
)

print("\nModel optimization process completed.")
print("The top performing weights asset configuration is saved as 'best_ser_hybrid_model.keras'")