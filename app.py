import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import streamlit as nn_st # Imported for UI rendering
import streamlit as st
import altair as alt
import pandas as pd
from funasr import AutoModel

# Set professional dashboard styling configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎯",
    layout="centered"
)

# =====================================================================
# 1. FIXED MODEL ARCHITECTURE MATCHING TRAINED CHECKPOINT
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
    def forward(self, x): 
        return self.network(x)

# =====================================================================
# 2. CACHED RESOURCE LOADERS (Prevents re-loading on UI interaction)
# =====================================================================
@st.cache_resource
def load_ssl_transformer():
    """Loads the native local emotion2vec feature extraction engine."""
    model_path = "./emotion2vec_model"
    # Disable updates to prevent internet access dependency during inference loops
    model = AutoModel(model=model_path, device="cpu", disable_update=True)
    return model

@st.cache_resource
def load_downstream_classifier():
    """Loads the optimized PyTorch speaker-independent classification adapter."""
    model_weights = "best_ser_ssl_model.pt"
    model = RobustEmotionClassifier(input_dim=768, num_classes=6)
    
    if os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location="cpu", weights_only=True))
    else:
        st.error(f"❌ Critical Error: '{model_weights}' weight file not found in directory!")
    model.eval()
    return model

# Initialize the global analytical backends
with st.spinner("🧠 Initializing Neural Network Engines (emotion2vec + PyTorch Head)..."):
    ssl_engine = load_ssl_transformer()
    classifier = load_downstream_classifier()

EMOTIONS_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# =====================================================================
# 3. STREAMLIT USER INTERFACE DESIGN
# =====================================================================
st.title("🎯 Speech Emotion Recognition (SER) Dashboard")
st.subheader("Tactical Self-Supervised Foundation Model Pipeline")
st.write("---")

st.markdown("""
### 📂 Upload Tactical Audio Workspace
Upload any **.wav** audio sample below. The pipeline will automatically parse the file through 
Alibaba's native `emotion2vec` runtime, extract context-aware 768-dimensional features, 
and deploy the speaker-independent neural classifier matrix.
""")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose an audio file (.wav format only)", type=["wav"])

if uploaded_file is not None:
    st.write("---")
    st.markdown("### 🎚️ Audio Audio Playback")
    # Display native audio widget
    st.audio(uploaded_file, format="audio/wav")
    
    # Save file temporarily to disk because funasr.generate requires a path string or array input
    temp_path = "temp_inference_sample.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("🧠 Slicing audio track and extracting true emotional context vectors..."):
        try:
            # 1. Protect GPU memory boundaries by enforcing a 5.0 second cap via librosa loading
            audio, sr = librosa.load(temp_path, sr=16000, duration=5.0)
            
            if len(audio) == 0:
                st.error("❌ Error: Uploaded audio track appears corrupted or completely empty.")
                st.stop()
                
            # 2. Compute the raw frame feats out-of-the-box
            res = ssl_engine.generate(input=audio, granularity="frame")
            
            # 3. Handle mean-pooling down to flat global feature array
            if len(res) > 0 and "feats" in res[0]:
                embedding = np.mean(res[0]["feats"], axis=0)
            else:
                st.error("❌ Failure: Native engine failed to map features from this audio file structure.")
                st.stop()
                
            # 4. Format sample array into torch tensor framework
            X_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0) # Shape: [1, 768]
            
            # 5. Execute downstream forward propagation
            with torch.no_grad():
                raw_logits = classifier(X_tensor)
                # Compute raw probabilities utilizing standard Softmax transformation
                probabilities = torch.softmax(raw_logits, dim=1).squeeze(0).numpy()
                
            # Clean up the local storage asset safely
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Map top prediction index metrics
            top_class_idx = np.argmax(probabilities)
            predicted_emotion = EMOTIONS_LIST[top_class_idx].upper()
            confidence_score = probabilities[top_class_idx] * 100
            
            # =====================================================================
            # 4. RENDER GRAPHICAL INFERENCE METRICS
            # =====================================================================
            st.write("---")
            st.markdown("### 📊 Neural Network Diagnostic Output")
            
            # Highlight final classified emotion using clean semantic metric displays
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Classified Emotion Layer", value=predicted_emotion)
            with col2:
                st.metric(label="🔒 Confidence Level Matrix", value=f"{confidence_score:.2f}%")
                
            # Build an elegant horizontal performance visualization leveraging Altair chart engines
            chart_data = pd.DataFrame({
                'Emotion Class': [e.capitalize() for e in EMOTIONS_LIST],
                'Probability Matrix': probabilities
            })
            
            bar_chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Probability Matrix:Q', axis=alt.Axis(format='%'), title="Classification Probability"),
                y=alt.Y('Emotion Class:N', sort='-x', title="Class Target"),
                color=alt.condition(
                    alt.datum['Probability Matrix'] == probabilities[top_class_idx],
                    alt.value('#1f77b4'), # Distinct highlight code for winning key
                    alt.value('#d3d3d3')  # Muted grey formatting for remaining elements
                )
            ).properties(height=300)
            
            st.altair_chart(bar_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Runtime pipeline failure during forward-pass optimization loops: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

st.write("---")
st.caption("💻 Developed via speaker-independent multi-corpus benchmarking protocols for DEAL, DRDO, Dehradun.")