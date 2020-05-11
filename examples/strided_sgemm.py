import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a = torch.randn(960, 1024).cuda()
b = torch.randn(1024, 1024).cuda()
aa, bb = a.clone().detach(), b.clone().detach()

a = F.pad(a, (0, 32)).narrow(-1, 0, 1024)
b = F.pad(b, (0, 32)).narrow(-1, 0, 1024)
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))

c = torch.matmul(a, b)
print("C size: {}, C stride: {}".format(c.size(), c.stride()))

cc = torch.matmul(aa, bb)

print("c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

