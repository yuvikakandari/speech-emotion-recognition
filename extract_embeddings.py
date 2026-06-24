import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 1. Force Device Mapping to your confirmed RTX 2050 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing Direct ModelScope Pipeline Engine on: {device}")

# 2. Load the model via explicit task-based pipeline (Bypasses Hugging Face 401 completely)
model_id = "iic/emotion2vec_base_v2"
inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model=model_id,
    model_revision="v2.0.4",
    device=device
)

DATASET_DIR = r"C:\Users\Yuvika\Desktop\DRDO speech-emotion-recognition\data"
OUTPUT_DIR = "./emotion2vec_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🧠 COMMENCING DIRECT EMOTION2VEC EMBEDDING EXTRACTION...")

# 3. Stream through directories actor-by-actor
for actor_folder in os.listdir(DATASET_DIR):
    actor_path = os.path.join(DATASET_DIR, actor_folder)
    if os.path.isdir(actor_path):
        print(f"\nProcessing vectors for: {actor_folder}")
        
        for filename in tqdm(os.listdir(actor_path)):
            if filename.endswith(".wav"):
                wav_path = os.path.join(actor_path, filename)
                
                try:
                    # Native emotion2vec requires a strict 16000Hz sampling layer
                    audio, sr = librosa.load(wav_path, sr=16000)
                    
                    # Generate feature maps using the direct pipeline execution context
                    res = inference_pipeline(audio)
                    
                    # Extract the global pooled semantic embedding vector (768 dimensions)
                    # Bypasses manual feature extraction processing loops!
                    if isinstance(res, list) and len(res) > 0 and 'feats' in res[0]:
                        embedding = np.array(res[0]['feats'], dtype=np.float32)
                    else:
                        continue
                    
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

print(f"\n✅ Extraction completed successfully! Feature matrices saved to: {OUTPUT_DIR}")