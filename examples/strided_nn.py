import torch
import torch.nn as nn
import torch.nn.functional as F

class StridedLinear(nn.Module):
	def __init__(self, emb_size, hidden_size):
		super().__init__()
		self.lin = nn.Linear(emb_size, hidden_size)

	def forward(self, x):
#		x = F.pad(x, (0, stride)).narrow(-1, 0, x.shape[-1])
#		for p in self.lin.parameters():
#			p.data = F.pad(p.data, (0, stride)).narrow(-1, 0, p.data.shape[-1])
		return self.lin(x)

bs, seq_len, emb_size, hidden_size, stride = 12, 512, 300, 1024, 32
x = torch.randn(seq_len, emb_size).cuda()
m = StridedLinear(emb_size, hidden_size).cuda()

mm = torch.jit.trace(m.forward, x)

print(mm.graph)
