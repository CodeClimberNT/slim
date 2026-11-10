import torch

X = torch.randn(5, 3)

y_pred = [
    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
    torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]),
    torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]),
    torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]),
    torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3]),
]

M = torch.stack(y_pred, dim=0)

for row in range(M.size()[0]):
    print(M[row])
