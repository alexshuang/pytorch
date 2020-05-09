import torch
import torch.nn as nn

a = torch.randn(512, 300 + 32).cuda()
a = a.narrow(-1, 0, 300)
#a = a.set_(a.storage(), 0, (512, 300), a.stride())
b = torch.randn(300, 1024 + 32).cuda()
b = b.narrow(-1, 0, 1024)
#b = b.set_(b.storage(), 0, (300, 1024), b.stride())
print(a.stride(), b.stride())
c =  torch.matmul(a, b)
print(c.shape, c.stride())
