import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import permutations, product
import math
from utils.general import prod
from torch.nn.functional import softmax
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        return gelu(x)


def gelu(x):
    return x * (1 + torch.erf(x) / math.sqrt(2)) / 2


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = (preds.t() == target).sum().float()
    return correct / (target.size(0) * target.size(1))


def symmetrize(x, es, size):
    # x: [batch, E]
    tmp = torch.zeros(x.size(0), size, size).long()
    if x.is_cuda:
        tmp = tmp.cuda()
    row, col = es
    tmp[:, row, col] = x
    rate = (tmp != tmp.transpose(1, 2)).sum().float() / (x.shape[0] * x.shape[1])
    return rate


def sym_transpose(x, es, size):
    # x: [E, batch, K]
    tmp = torch.zeros(size, size, x.size(1), x.size(-1))
    tmp.requires_grad = True
    if x.is_cuda:
        tmp = tmp.cuda(x.device)
        es = es.cuda(x.device)
    row, col = es
    tmp[row, col, :, :] = x
    tmp = tmp.transpose(0, 1).contiguous()
    z = tmp[row, col, :, :]
    return z


def sym_transposev2(x, size):
    idx = torch.arange(size * (size - 1))
    ii = idx // (size - 1)
    jj = idx % (size - 1)
    jj = jj * (jj < ii).long() + (jj + 1) * (jj >= ii).long()
    index = jj * (size - 1) + ii * (ii < jj).long() + (ii - 1) * (ii > jj).long()
    return x[index]


def sym(x, es, size):
    # x: [E, batch, K]
    tmp = torch.zeros(size, size, x.size(1), x.size(-1))
    tmp.requires_grad = True
    if x.is_cuda:
        tmp = tmp.cuda(x.device)
    row, col = es
    tmp[row, col, :, :] = x
    tmp = (tmp + tmp.transpose(0, 1).contiguous()) / 2
    z = tmp[row, col, :, :]
    return z


def min_max(x, a=0, b=1):
    # min-max normalization
    # scale to [a, b]
    xmin = x.min()
    xmax = x.max()
    z = (x - xmin) / (xmax - xmin)
    return (b - a) * z + a


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return (y / tau).sigmoid()


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return (uniform + eps).log() - (1 - uniform + eps).log()


def my_bn(x, bn):
    # x: tensor, bn: BatchNorm1d
    shape = x.shape
    z = x.view(prod(shape[:-1]), shape[-1])
    z = bn(z)
    z = z.view(*shape)
    return z


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return - (- (U.relu() + eps).log().clamp(max=0.) + eps).log()


def sample_gumbel_max(logits, eps=1e-10, one_hot=False):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    ms, index = y.max(-1, keepdim=True)
    es = (y >= ms) if one_hot else index
    return es


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    return (y / tau).softmax(-1)


def inv_one_hot(x):
    idx = x.nonzero(as_tuple=True)
    h = x.sum(-1).view(1, -1)
    h = h * idx[-1]
    h = h.view(*x.shape[:-1])
    return h


def my_cdist(x, y, p=1):
    assert (x.shape == y.shape).all()
    n = len(x.shape)
    xx = x.unsqueeze(n-2)
    yy = y.unsqueeze(n-1)
    xx, yy = torch.broadcast_tensors(xx, yy)
    z = xx - yy
    return z.norm(dim=-1, p=p)


def sampler(x, batch):
    index = np.random.choice(range(len(x)), batch, replace=False)
    return x[index]
