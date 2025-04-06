import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
    print("GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')  # Fall back to CPU
    print("GPU not available. Using CPU.")
