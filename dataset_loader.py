import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

# ---------------------------------------------------
# DATASET PATHS
# ---------------------------------------------------

DATASET_PATH = "data/raw_dataset"

# Folder where extracted audio from videos will be saved
EXTRACTED_AUDIO_PATH = "data/extracted_audio"

# Create folder if it doesn't exist
os.makedirs(EXTRACTED_AUDIO_PATH, exist_ok=True)

# ---------------------------------------------------
# EMOTION LABEL MAPPING
# ---------------------------------------------------

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ---------------------------------------------------
# LISTS TO STORE DATA
# ---------------------------------------------------

paths = []
emotions = []
modalities = []

# ---------------------------------------------------
# TRAVERSE ENTIRE DATASET
# ---------------------------------------------------

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        try:

            # ---------------------------------------------------
            # CHECK FILE TYPE
            # ---------------------------------------------------

            if file.endswith(".wav") or file.endswith(".mp4"):

                # Full file path
                file_path = os.path.join(root, file)

                # ---------------------------------------------------
                # SPLIT FILENAME
                # Example:
                # 02-01-06-01-02-01-12.mp4
                # ---------------------------------------------------

                parts = file.split("-")

                # ---------------------------------------------------
                # EXTRACT MODALITY
                # ---------------------------------------------------

                modality_code = parts[0]

                if modality_code == "01":
                    modality = "full_av"

                elif modality_code == "02":
                    modality = "video_only"

                elif modality_code == "03":
                    modality = "audio_only"

                else:
                    modality = "unknown"

                # ---------------------------------------------------
                # EXTRACT EMOTION LABEL
                # ---------------------------------------------------

                emotion_code = parts[2]

                emotion = emotion_map[emotion_code]

                # ---------------------------------------------------
                # HANDLE AUDIO FILES
                # ---------------------------------------------------

                if file.endswith(".wav"):

                    final_audio_path = file_path

                # ---------------------------------------------------
                # HANDLE VIDEO FILES
                # EXTRACT AUDIO FROM VIDEO
                # ---------------------------------------------------

                elif file.endswith(".mp4"):

                    # Create output audio filename
                    audio_filename = file.replace(".mp4", ".wav")

                    # Output path
                    final_audio_path = os.path.join(
                        EXTRACTED_AUDIO_PATH,
                        audio_filename
                    )

                    # Extract only if not already extracted
                    if not os.path.exists(final_audio_path):

                        video = VideoFileClip(file_path)

                        video.audio.write_audiofile(
                            final_audio_path,
                            codec='pcm_s16le',
                            logger=None
                        )

                # ---------------------------------------------------
                # STORE DATA
                # ---------------------------------------------------

                paths.append(final_audio_path)

                emotions.append(emotion)

                modalities.append(modality)

                print(f"Processed: {file}")

        except Exception as e:

            print(f"Error processing {file}: {e}")

# ---------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------

df = pd.DataFrame({
    "path": paths,
    "emotion": emotions,
    "modality": modalities
})

# ---------------------------------------------------
# SAVE DATAFRAME
# ---------------------------------------------------

df.to_csv("dataset.csv", index=False)

# ---------------------------------------------------
# DISPLAY INFORMATION
# ---------------------------------------------------

print("\nDataset Loaded Successfully")

print("\nTotal Samples:", len(df))

print("\nEmotion Distribution:\n")
print(df["emotion"].value_counts())

print("\nModality Distribution:\n")
print(df["modality"].value_counts()) 

print("\nFirst 5 Rows:\n")
print(df.head())
