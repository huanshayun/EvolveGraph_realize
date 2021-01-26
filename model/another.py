import torch
import

def get_idx():
    batch, node, step, dim = 16, 10, 50, 2
    classes = 5
    hidden = 256
    # node class
    x = torch.randint(0, classes, (batch, node))
    # node state
    y = torch.rand(batch, node, step, dim)
    # class specific mlps
    mlps = torch.nn.ModuleList([torch.nn.Linear(dim, hidden) for _ in range(classes)])
    # index
    idx = [x == i for i in range(classes)]  # for each category, determine whether an element in X belongs to that category
    hs = [mlps[i](y[j]) for i, j in zip(range(classes), idx)]
    h = torch.zeros(batch, node, step, hidden)
    for i, j in zip(range(classes), idx):
        h[j] = hs[i]
    pass


if __name__ == "__main__":
    get_idx()
