import torch
print("CUDA Available:", torch.cuda.is_available())  # Should print True
print("CUDA Version:", torch.version.cuda)  # Should match NVIDIA-SMI (12.6)
print("PyTorch Build:", torch.__version__)  # Should show 'cu126' or similar
print("Available Devices:", torch.cuda.device_count())  # Should be > 0
