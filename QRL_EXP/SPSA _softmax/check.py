import torch
import torch.nn as nn
import numpy as np

a = torch.tensor([[1,2,3,4,-float('inf')]])

prob = nn.Softmax(dim=1)


print(prob(a).squeeze(0).tolist())

b =prob(a).squeeze(0)/sum(prob(a).squeeze(0))

print(np.random.choice([0,1,2,3,5],p=b))