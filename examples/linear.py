import torch
import torch.nn as nn

x = torch.randn(512, 300).cuda()
l = nn.Linear(300, 1000, False).cuda()
y = l(x)
print(y.size(), y.stride())
