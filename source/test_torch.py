import torch
import torch.nn.functional as F
import torch.nn as nn

print("torch version:", torch.__version__)

x = torch.randn(2, 3, 4, 4)
y = torch.randn(2, 3, 4, 4)

print("l1_loss:", F.l1_loss(x, y))  # 这里应该不会再报 "mean is not a valid value"

m = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
sd = m.state_dict()
print("state_dict keys:", list(sd.keys()))
