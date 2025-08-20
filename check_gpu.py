import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
    print("Memory allocated:", torch.cuda.memory_allocated(0))
    print("Memory cached:", torch.cuda.memory_reserved(0))
else:
    print("Running on CPU")

# Test tensor creation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create a test tensor
x = torch.randn(10, 10).to(device)
print(f"Test tensor device: {x.device}")