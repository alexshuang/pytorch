import torch
import torch.nn as nn
import torch.nn.functional as F

stride = 32

x = torch.randn(12, 500, 300).half().cuda()
l = nn.Linear(300, 1000).half().cuda()
xx = torch.clone(x).detach()
w, b = torch.clone(l.weight).detach(), torch.clone(l.bias).detach()

x = F.pad(x, (0, stride)).narrow(-1, 0, x.shape[-1])
for o in l.parameters():
	if o.data.dim() > 1:
		o.data = F.pad(o.data, (0, stride)).narrow(-1, 0, o.data.shape[-1])

print("x: ", x.size(), x.stride())
print("w: ", l.weight.size(), l.weight.stride())
print("b: ", l.bias.size(), l.bias.stride())
y = l(x)
print("y: ", y.size(), y.stride())

yy = xx @ w.T + b
print("yy: ", yy.size(), yy.stride())

print("y == yy: {}".format(torch.allclose(y.float().cpu(), yy.float().cpu())))
