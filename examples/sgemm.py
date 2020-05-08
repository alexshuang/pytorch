import torch
import torch.nn as nn

#lin = nn.Linear(20, 50, bias=False).half().cuda()
#x = torch.randn(10, 20).half().cuda()
#print(torch.jit.trace(lin, x).graph)
#y = lin(x)
#print(y.shape)

a = torch.randn(512, 300).cuda()
b = torch.randn(300, 1024).cuda()
c =  torch.matmul(a, b)
print(c.shape)
