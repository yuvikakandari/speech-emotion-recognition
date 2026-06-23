import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. SPEAKER-INDEPENDENT 3-WAY STRATIFICATION LOGIC
# =====================================================================
EMBEDDINGS_DIR = "./emotion2vec_embeddings"

all_files = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npz")]

# Parse out speakers and emotions to run accurate splitting loops
speakers = []
labels = []
for fpath in all_files:
    data = np.load(fpath)
    speakers.append(str(data['speaker']))
    labels.append(str(data['label']))

unique_speakers = list(set(speakers))

# Step A: Carve out a clean 10% pool of actors for the blind final test split
train_val_speakers, test_speakers = train_test_split(
    unique_speakers, test_size=0.10, random_state=42
)
# Step B: Take the remaining 90% and carve out a pure 10% pool for validation checks
train_speakers, val_speakers = train_test_split(
    train_val_speakers, test_size=0.1111, random_state=42
)

# Filter files into completely isolated physical lists
train_files = [f for f, spk in zip(all_files, speakers) if spk in train_speakers]
val_files = [f for f, spk in zip(all_files, speakers) if spk in val_speakers]
test_files = [f for f, spk in zip(all_files, speakers) if spk in test_speakers]

print(f"🧬 Splitting Integrity: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test.")

# Encode target categorical text labels to numeric vectors (ANG->0, SAD->1, etc.)
le = LabelEncoder()
le.fit(labels)

# =====================================================================
# 2. PYTORCH CUSTOM DATA UTILITIES
# =====================================================================
class SERVectorDataset(Dataset):
    def __init__(self, file_list, encoder):
        self.file_list = file_list
        self.encoder = encoder
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        X = torch.tensor(data['embedding'], dtype=torch.float32)
        # Handle structural shape squeezing from pooling layers
        if X.ndim > 1: X = torch.mean(X, dim=0) 
        y = torch.tensor(self.encoder.transform([str(data['label'])])[0], dtype=torch.long)
        return X, y

train_loader = DataLoader(SERVectorDataset(train_files, le), batch_size=64, shuffle=True)
val_loader = DataLoader(SERVectorDataset(val_files, le), batch_size=64, shuffle=False)

# =====================================================================
# 3. DOWNSTREAM CLASSIFICATION HEAD NETWORK ARCHITECTURE
# =====================================================================
class EmotionClassifierHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=len(le.classes_)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EmotionClassifierHead().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# =====================================================================
# 4. VELOCITY DRIVEN TRAINING ENGINE WITH EARLY STOPPING
# =====================================================================
best_val_loss = float('inf')
patience, patience_counter = 7, 0

print("\n🏁 COMMENCING DOWNSTREAM OPERATIONAL TRAINING SEQUENCE...")
for epoch in range(1, 40):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
        
    train_loss = total_loss / total
    train_acc = correct / total
    
    # Run structural validation validation pass
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            v_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            v_total += y_batch.size(0)
            v_correct += predicted.eq(y_batch).sum().item()
            
    val_loss = v_loss / v_total
    val_acc = v_correct / v_total
    
    print(f"Epoch {epoch:02d} -> Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% || Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Track Early Stopping checkpoints cleanly
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_ssl_classifier.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"🛑 Early stopping kicked in. Reverting weight mappings back to historical validation floor.")
            break