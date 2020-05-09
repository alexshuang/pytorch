import torch
import torch.nn as nn

a = torch.randn(512, 300).cuda()
b = torch.randn(300, 1024).cuda()
c =  torch.matmul(a, b)
print(c.shape)
