import torch

# Fix the random seed
torch.manual_seed(42)

# Random tensor of size (2, 3)
tensor = torch.rand(2, 3)

print(tensor)