import torch
import torch.nn as nn
import torch.nn.functional as F

stride = 32

x = torch.randn(12, 500, 300).cuda()
layers = [nn.Linear(300, 1000), nn.Linear(1000, 800)]
m = nn.Sequential(*layers).cuda()

x = F.pad(x, (0, stride)).narrow(-1, 0, x.shape[-1])
for o in m.parameters():
	if o.data.dim() > 1:
		o.data = F.pad(o.data, (0, stride)).narrow(-1, 0, o.data.shape[-1])

print("x: ", x.size(), x.stride())
y = m(x)
print("y: ", y.size(), y.stride())
