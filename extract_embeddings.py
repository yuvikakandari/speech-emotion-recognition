import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

# 1. Map to your newly confirmed RTX 2050 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing Hugging Face Extraction Engine on: {device}")

# 2. Pull directly from the global Hugging Face distribution mirror
model_id = "iic/emotion2vec_base_v2"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

DATASET_DIR = r"C:\Users\Yuvika\Desktop\DRDO speech-emotion-recognition\data"
OUTPUT_DIR = "./emotion2vec_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🧠 COMMENCING HUGGING FACE SSL EMBEDDING EXTRACTION...")
for actor_folder in os.listdir(DATASET_DIR):
    actor_path = os.path.join(DATASET_DIR, actor_folder)
    if os.path.isdir(actor_path):
        print(f"Processing vectors for: {actor_folder}")
        
        for filename in tqdm(os.listdir(actor_path)):
            if filename.endswith(".wav"):
                wav_path = os.path.join(actor_path, filename)
                
                try:
                    # Load audio at native 16kHz
                    audio, sr = librosa.load(wav_path, sr=16000)
                    
                    # Preprocess raw wave array through the model's feature layer
                    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    # Extract final hidden state representations and pool them globally
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
                    
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

print(f"✅ Extraction completed successfully! Target folder: {OUTPUT_DIR}")