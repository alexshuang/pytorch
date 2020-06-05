import torch
import torch.nn as nn
import torch.nn.functional as F

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
print(c[0, 0, 0, :5], cc[0, 0, 0, :5])
#print("NN: c == cc: {}".format(torch.allclose(c, cc)))

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
cc = aa @ bb.transpose(2, 3)
print(c[0, 0, 0, :5], cc[0, 0, 0, :5])
#print("NT: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

# TN
a = torch.randn(12, 6, 300, 500).cuda()
b = torch.randn(12, 6, 300, 1000).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a.transpose(-1, -2) @ b
print("TN: C size: {}, C stride: {}".format(c.size(), c.stride()))
cc = aa.transpose(-1, -2) @ bb
print(c[0, 0, 0, :5], cc[0, 0, 0, :5])
#print("TN: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))

# TT
a = torch.randn(12, 6, 300, 500).cuda()
b = torch.randn(12, 6, 1000, 300).cuda()
aa, bb = a.clone().detach(), b.clone().detach()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = F.pad(b, (0, 32)).narrow(-1, 0, b.size(-1))
print("A size: {}, A stride: {}".format(a.size(), a.stride()))
print("B size: {}, B stride: {}".format(b.size(), b.stride()))
c = a.transpose(-1, -2) @ b.transpose(-1, -2)
print("TT: C size: {}, C stride: {}".format(c.size(), c.stride()))
cc = aa.transpose(-1, -2) @ bb.transpose(-1, -2)
print(c[0, 0, 0, :5], cc[0, 0, 0, :5])
#print("TT: c == cc: {}".format(torch.allclose(c.cpu(), cc.cpu())))
