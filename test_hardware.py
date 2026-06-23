import torch

print("=== PyTorch Hardware Check ===")
gpu_available = torch.cuda.is_available()
print(f"Is NVIDIA GPU acceleration available? {gpu_available}")

if gpu_available:
    print(f"Active GPU Device Name: {torch.cuda.get_device_name(0)}")
    
    # Configure your pipeline's target processing chip dynamically
    device = torch.device("cuda")
    print("🚀 Status: Success! PyTorch is ready to offload matrix operations to your RTX card.")
else:
    device = torch.device("cpu")
    print("⚠️ Status: PyTorch cannot find the CUDA runtime. Defaulting to CPU.")