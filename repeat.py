
import torch


f = torch.rand(8, 2)

print(f.shape)

print(f.repeat(8, 64, 2, 1).shape)
