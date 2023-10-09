import torch
from progen.model import ProGen

x = torch.randint(0, 100, (1, 1024))
model = ProGen(num_tokens=100, dim=512, seq_len=1024, depth=6)
outputs = model(x)
print(outputs)