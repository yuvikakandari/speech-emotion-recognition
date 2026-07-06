import os
path = r"C:\Users\Yuvika\Desktop\DRDO speech-emotion-recognition\datasets"
for root, dirs, files in os.walk(path):
    wavs = [f for f in files if f.endswith('.wav')]
    if wavs:
        print(f"📁 Folder: {os.path.basename(root)}")
        print(f"📄 Sample Filenames: {wavs[:3]}\n")