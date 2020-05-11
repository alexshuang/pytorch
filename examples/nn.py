import torch
import torch.nn as nn
import torch.nn.functional as F

bs, seq_len, emb_size, hidden_size, stride = 12, 512, 300, 1024, 32
x = torch.randn(seq_len, emb_size).cuda()
model = nn.Linear(emb_size, hidden_size, bias=False).cuda()

y = model(x)
