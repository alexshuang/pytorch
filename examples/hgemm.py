import torch
import torch.nn as nn

#a = torch.randn(512, 300, dtype=torch.float16).cuda()
#b = torch.randn(300, 1024, dtype=torch.float16).cuda()
a = torch.randn(512, 300).half().cuda()
b = torch.randn(300, 1024).half().cuda()
c =  torch.matmul(a, b)
print(c.shape)
