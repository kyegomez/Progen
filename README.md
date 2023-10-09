[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Progen
Implementation of Progen in Pytorch, from the paper "ProGen: Language Modeling for Protein Generation"

GPT for proteins sequences

[Paper Link](https://arxiv.org/pdf/2004.03497.pdf)

# Appreciation
* Lucidrains
* Agorians

# Install
`pip install progen-torch`

# Usage
```python
import torch
from progen.model import ProGen

x = torch.randint(0, 100, (1, 1024))
model = ProGen(num_tokens=100, dim=512, seq_len=1024, depth=6)
outputs = model(x)
print(outputs)


```

# Architecture

# Todo


# License
MIT

# Citations

