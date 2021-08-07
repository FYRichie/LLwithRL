import torch
import numpy as np
a = []
for i in range(10):
    a.append(torch.Tensor([i/10]))
print(np.shape(list(map(lambda x: x.item(), a))))
a = torch.stack(a)
b = []
for i in range(10):
    b.append(torch.Tensor([i]))
b = torch.stack(b)
print(np.shape((a * b).detach().numpy()))