import torch

print("=========================================")
print("   CUDA ACCELERATION INTERFACE CHECK   ")
print("=========================================")
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA (GPU Support) active? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Total Available GPUs: {torch.cuda.device_count()}")
    print(f"Target Device Name: {torch.cuda.get_device_name(0)}")
    print("🚀 Status: Success! Operations will run via your NVIDIA RTX card.")
else:
    print("⚠️ Status: Running via standard CPU instructions.")
print("=========================================")