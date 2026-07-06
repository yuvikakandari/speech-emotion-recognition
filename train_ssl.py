import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# =====================================================================
# 1. FIXED PARSER ENGINE FOR MULTI-DATASETS (CREMA-D, RAVDESS, TESS)
# =====================================================================
EMBEDDINGS_DIR = "./emotion2vec_embeddings"
if not os.path.exists(EMBEDDINGS_DIR) or len(os.listdir(EMBEDDINGS_DIR)) == 0:
    print(f"❌ Error: The embeddings folder '{EMBEDDINGS_DIR}' is empty!")
    sys.exit()

all_files = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npz")]

# Explicit target mappings matching dataset specifications perfectly
RAVDESS_MAP = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear', '07': 'disgust'}
CREMA_MAP   = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
TESS_MAP    = {'ANGRY': 'angry', 'DISGUST': 'disgust', 'FEAR': 'fear', 'HAPPY': 'happy', 'NEUTRAL': 'neutral', 'SAD': 'sad'}

valid_files = []
mapped_labels = []

print("📂 Parsing multi-dataset embedding keys with perfect precision rules...")
for fpath in all_files:
    filename = os.path.basename(fpath).upper().replace(".NPZ", "")
    
    derived_emotion = None
    
    # CASE 1: RAVDESS Format (e.g., 03-01-01-01-01-01-01)
    if "-" in filename and len(filename.split("-")) >= 7:
        parts = filename.split("-")
        emotion_code = parts[2]
        if emotion_code in RAVDESS_MAP:
            derived_emotion = RAVDESS_MAP[emotion_code]

    # CASE 2: CREMA-D Format (e.g., 1001_DFA_ANG_XX)
    elif "_" in filename and ("ANG" in filename or "DIS" in filename or "FEA" in filename or "HAP" in filename or "NEU" in filename or "SAD" in filename):
        parts = filename.split("_")
        for p in parts:
            if p in CREMA_MAP:
                derived_emotion = CREMA_MAP[p]
                break

    # CASE 3: TESS Format (e.g., OAF_BACK_ANGRY or YAF_BAR_SAD)
    else:
        parts = filename.split("_")
        last_part = parts[-1] 
        if last_part in TESS_MAP:
            derived_emotion = TESS_MAP[last_part]
        elif last_part == "PS": 
            continue

    if derived_emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']:
        valid_files.append(fpath)
        mapped_labels.append(derived_emotion)

# =====================================================================
# 🚨 CRITICAL CHANGE: SPEAKER-DEPENDENT (GLOBAL RANDOM SPLIT)
# =====================================================================
# Instead of splitting by unique speaker groups, we shuffle ALL files globally.
# This matches the common academic approach mentioned by your senior.
train_files, test_files = train_test_split(valid_files, test_size=0.15, random_state=42, stratify=mapped_labels)
train_files, val_files = train_test_split(train_files, test_size=0.15, random_state=42, stratify=[mapped_labels[valid_files.index(f)] for f in train_files])

file_to_label_dict = dict(zip(valid_files, mapped_labels))

print("=====================================================================")
print("🧬 SPEAKER-DEPENDENT GLOBAL RANDOM 6-CLASS PIPELINE")
print("=====================================================================")
print(f"📊 Valid Files Found: {len(valid_files)} / {len(all_files)}")
print(f"📊 Splits -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

le = LabelEncoder()
le.fit(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'])

# =====================================================================
# 2. Z-SCORE NORMALIZATION FILTER
# =====================================================================
print("🧮 Fitting baseline training normalization scaler...")
train_features = []
for fpath in train_files:
    emb = np.load(fpath)['embedding'].flatten()
    train_features.append(emb)

scaler = StandardScaler()
scaler.fit(train_features)

class SERPipelineDataset(Dataset):
    def __init__(self, file_list, encoder, data_scaler, label_dict):
        self.file_list = file_list
        self.encoder = encoder
        self.scaler = data_scaler
        self.label_dict = label_dict
        
    def __len__(self): return len(self.file_list)
        
    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        X = np.load(fpath)['embedding'].flatten()
        X_norm = self.scaler.transform([X])[0]
        
        label_str = self.label_dict[fpath]
        y = torch.tensor(self.encoder.transform([label_str])[0], dtype=torch.long)
        return torch.tensor(X_norm, dtype=torch.float32), y

train_loader = DataLoader(SERPipelineDataset(train_files, le, scaler, file_to_label_dict), batch_size=64, shuffle=True)
val_loader = DataLoader(SERPipelineDataset(val_files, le, scaler, file_to_label_dict), batch_size=64, shuffle=False)
test_loader = DataLoader(SERPipelineDataset(test_files, le, scaler, file_to_label_dict), batch_size=64, shuffle=False)

# =====================================================================
# 3. CLASSIFIER ARCHITECTURE (768 DIMENSIONS NATIVE)
# =====================================================================
class RobustEmotionClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.network(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RobustEmotionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

# =====================================================================
# 4. TRAINING EXECUTION LOOP
# =====================================================================
best_val_loss = float('inf')
early_stop_counter = 0

print("\n🏁 NEURAL WORKFLOW INITIALIZED...")
for epoch in range(1, 40):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        outputs = model(X_b)
        loss = criterion(outputs, y_b)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_b.size(0)
        _, pred = outputs.max(1)
        total += y_b.size(0)
        correct += pred.eq(y_b).sum().item()
        
    train_loss, train_acc = total_loss / total, correct / total
    
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            v_loss += loss.item() * X_b.size(0)
            _, pred = outputs.max(1)
            v_total += y_b.size(0)
            v_correct += pred.eq(y_b).sum().item()
            
    val_loss, val_acc = v_loss / v_total, v_correct / v_total
    print(f"Epoch {epoch:02d} -> Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% || Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.1f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_ser_ssl_model_dependent.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 6:
            print("🛑 Early Stopping triggered!")
            break

# =====================================================================
# 5. TEST ACCURACY ASSESSMENT
# =====================================================================
print("\n🔒 TESTING BALANCED BLIND POOL...")
model.load_state_dict(torch.load("best_ser_ssl_model_dependent.pt", weights_only=True))
model.eval()

t_correct, t_total = 0, 0
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        outputs = model(X_b)
        _, pred = outputs.max(1)
        t_total += y_b.size(0)
        t_correct += pred.eq(y_b).sum().item()

print("=====================================================================")
print(f"🎯 FINAL SPEAKER-DEPENDENT ACCURACY: {(t_correct/t_total)*100:.2f}%")
print("=====================================================================")