import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NT
a = torch.randn(12, 6, 500, 300).cuda()
b = torch.randn(12, 6, 1000, 300).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a @ b.transpose(2, 3)
print("NT: C size: {}, C stride: {}".format(c.size(), c.stride()))
print("A size: {}, A stride: {}".format(aa.size(), aa.stride()))
print("B size: {}, B stride: {}".format(bb.size(), bb.stride()))
cc = aa @ bb.transpose(2, 3)
print("NT: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

'''
# NN
a = torch.randn(12, 6, 500, 300).cuda()
b = torch.randn(12, 6, 300, 1000).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a @ b
print("NN: C size: {}, C stride: {}".format(c.size(), c.stride()))
cc = aa @ bb
print("NN: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

# TN
a = torch.randn(12, 6, 300, 500).cuda()
b = torch.randn(12, 6, 300, 1000).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a.T @ b
print("TN: C size: {}, C stride: {}".format(c.size(), c.stride()))
cc = aa.T @ bb
print("TN: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

# TT
a = torch.randn(12, 6, 300, 500).cuda()
b = torch.randn(12, 6, 1000, 300).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a.T @ b.T
print("TT: C size: {}, C stride: {}".format(c.size(), c.stride()))
cc = aa.T @ bb.T
print("TT: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))
'''
