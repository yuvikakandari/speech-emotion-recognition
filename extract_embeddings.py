import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from funasr import AutoModel

# 1. Force Device Mapping to your RTX 2050
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing Extraction Interface on target device: {device}")

# 2. Download and load emotion2vec universal model weights via FunASR
# This model converts raw speech waveforms directly into compact emotion matrices
model = AutoModel(
    model="iic/emotion2vec_base_v2", 
    model_revision="v2.0.4",
    device=device
)

# Replace this string path with your local dataset folder directory
DATASET_DIR = r"C:\Users\Yuvika\Desktop\DRDO speech-emotion-recognition\data"
OUTPUT_DIR = "./emotion2vec_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Stream through directories actor-by-actor
print("🧠 COMMENCING TRANSFORMATION EMBEDDING EXTRACTION...")
for actor_folder in os.listdir(DATASET_DIR):
    actor_path = os.path.join(DATASET_DIR, actor_folder)
    if os.path.isdir(actor_path):
        print(f"Processing structural audio vectors for: {actor_folder}")
        
        # Track maps dynamically
        for filename in tqdm(os.listdir(actor_path)):
            if filename.endswith(".wav"):
                wav_path = os.path.join(actor_path, filename)
                
                try:
                    # Native emotion2vec requires a strict 16000Hz sampling layer
                    audio, sr = librosa.load(wav_path, sr=16000)
                    
                    # Compute forward pass representation through the frozen SSL network
                    with torch.no_grad():
                        res = model.generate(input=audio, sampling_rate=16000)
                        
                    # Extract the global pooled semantic embedding vector (768 dimensions)
                    # This completely replaces manual MFCC tracking math
                    embedding = res[0]['feats']
                    
                    # Save embedding and isolate labels from the file structure name
                    # Target structure format: [ActorID]_[TextID]_[EMOTION]_[Repetition]
                    label = filename.split('_')[2] if len(filename.split('_')) >= 3 else "UNKNOWN"
                    
                    output_filename = filename.replace(".wav", ".npz")
                    np.savez(
                        os.path.join(OUTPUT_DIR, output_filename), 
                        embedding=embedding, 
                        label=label,
                        speaker=actor_folder
                    )
                except Exception as e:
                    continue

print(f"✅ Extraction completed! All deep arrays securely saved to: {OUTPUT_DIR}")