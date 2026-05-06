import torch
from models.generator import RMEGenerator

model = RMEGenerator(input_nc=2, output_nc=1)
x = torch.randn(1, 2, 64, 64)
y = model(x)
print("Output shape:", y.shape)
