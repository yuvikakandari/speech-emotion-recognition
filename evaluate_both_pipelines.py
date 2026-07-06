import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. FIXED MULTI-DATASET PATH PARSER
# =====================================================================
EMBEDDINGS_DIR = "./emotion2vec_embeddings"
if not os.path.exists(EMBEDDINGS_DIR) or len(os.listdir(EMBEDDINGS_DIR)) == 0:
    print(f"❌ Error: The embeddings folder '{EMBEDDINGS_DIR}' is empty!")
    sys.exit()

all_files = [os.path.join(EMBEDDINGS_DIR, f) for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npz")]

RAVDESS_MAP = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fear', '07': 'disgust'}
CREMA_MAP   = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
TESS_MAP    = {'ANGRY': 'angry', 'DISGUST': 'disgust', 'FEAR': 'fear', 'HAPPY': 'happy', 'NEUTRAL': 'neutral', 'SAD': 'sad'}

valid_files = []
speakers = []
mapped_labels = []

for fpath in all_files:
    filename = os.path.basename(fpath).upper().replace(".NPZ", "")
    derived_emotion = None
    speaker_id = "UNKNOWN"
    
    if "-" in filename and len(filename.split("-")) >= 7:
        parts = filename.split("-")
        emotion_code = parts[2]
        speaker_id = f"RAVDESS_ACTOR_{parts[6]}"
        if emotion_code in RAVDESS_MAP: derived_emotion = RAVDESS_MAP[emotion_code]
    elif "_" in filename and any(x in filename for x in ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]):
        parts = filename.split("_")
        speaker_id = f"CREMA_ACTOR_{parts[0]}"
        for p in parts:
            if p in CREMA_MAP:
                derived_emotion = CREMA_MAP[p]
                break
    else:
        parts = filename.split("_")
        speaker_id = f"TESS_SPEAKER_{parts[0]}"
        last_part = parts[-1]
        if last_part in TESS_MAP: derived_emotion = TESS_MAP[last_part]

    if derived_emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']:
        valid_files.append(fpath)
        speakers.append(speaker_id)
        mapped_labels.append(derived_emotion)

file_to_label_dict = dict(zip(valid_files, mapped_labels))

le = LabelEncoder()
EMOTIONS_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
le.fit(EMOTIONS_LIST)

# =====================================================================
# 2. CHOOSE EMBEDDING LOADER DATASET CLASS
# =====================================================================
class SEREvaluationDataset(Dataset):
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

# =====================================================================
# 3. FIXED CLASSIFIER ARCHITECTURE (MATCHING TRAINED WEIGHTS)
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

def generate_report_and_matrix(model_path, test_loader, title, save_img_name):
    """Loads a specific model checkpoint and computes detailed metrics."""
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model weights file '{model_path}' not found! Skipping evaluation.")
        return
        
    model = RobustEmotionClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.numpy())
            
    # Print the crisp text report
    print(f"\n=====================================================================")
    print(f"📊 {title.upper()} PERFORMANCE EVALUATION")
    print(f"=====================================================================")
    print(classification_report(all_targets, all_preds, target_names=EMOTIONS_LIST, digits=4))
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS_LIST, yticklabels=EMOTIONS_LIST,
                cbar=True, square=True, annot_kws={"size": 11, "weight": "bold"})
    
    plt.title(f'Confusion Matrix - {title}', fontsize=13, weight='bold', pad=15)
    plt.ylabel('Actual Label', fontsize=11, weight='bold')
    plt.xlabel('Predicted Label', fontsize=11, weight='bold')
    plt.tight_layout()
    plt.savefig(save_img_name, dpi=300)
    plt.close()
    print(f"💾 Confusion Matrix plot successfully saved as: {save_img_name}")

# =====================================================================
# 4. EXECUTE PIPELINE EVALUATIONS
# =====================================================================

# --- RUN 1: SPEAKER-INDEPENDENT ASSESSMENT ---
print("\n🔒 Splitting data via Speaker-Independent constraints...")
unique_speakers = sorted(list(set(speakers)))
train_spk, test_spk = train_test_split(unique_speakers, test_size=0.15, random_state=42)
train_spk, val_spk = train_test_split(train_spk, test_size=0.15, random_state=42)

ind_train_files = [f for f, s in zip(valid_files, speakers) if s in train_spk]
ind_test_files = [f for f, s in zip(valid_files, speakers) if s in test_spk]

ind_train_features = [np.load(fpath)['embedding'].flatten() for fpath in ind_train_files]
scaler_ind = StandardScaler().fit(ind_features_train := ind_train_features)

ind_test_loader = DataLoader(SEREvaluationDataset(ind_test_files, le, scaler_ind, file_to_label_dict), batch_size=64, shuffle=False)
generate_report_and_matrix("best_ser_ssl_model.pt", ind_test_loader, "SSL Speaker-Independent", "confusion_matrix_independent.png")

# --- RUN 2: SPEAKER-DEPENDENT ASSESSMENT ---
print("\n🔓 Splitting data via Speaker-Dependent constraints...")
dep_train_files, dep_test_files = train_test_split(valid_files, test_size=0.15, random_state=42, stratify=mapped_labels)

dep_train_features = [np.load(fpath)['embedding'].flatten() for fpath in dep_train_files]
scaler_dep = StandardScaler().fit(dep_train_features)

dep_test_loader = DataLoader(SEREvaluationDataset(dep_test_files, le, scaler_dep, file_to_label_dict), batch_size=64, shuffle=False)
generate_report_and_matrix("best_ser_ssl_model_dependent.pt", dep_test_loader, "SSL Speaker-Dependent", "confusion_matrix_dependent.png")

print("\n=====================================================================")
print("✅ Evaluation complete! Add the printed text blocks and saved .png figures to your report chapters.")
print("=====================================================================\n")