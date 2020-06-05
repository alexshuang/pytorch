import torch
import torch.nn as nn
import torch.nn.functional as F

# NN
a = torch.randn(12, 6, 500, 300).cuda()
a = F.pad(a, (0, 32)).narrow(-1, 0, a.size(-1))
b = a.contiguous()
print(b.size(), b.stride())
