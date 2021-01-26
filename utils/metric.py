import torch
import config as cfg

def entropy(p, q, eps=1e-16):
    return (p * (q.relu() + eps).log()).sum()


def kl_divergence(p, q, eps=1e-16):
    return entropy(p, p, eps) - entropy(p, q, eps)


def nll_gaussian(preds, target, variance):
    # target: [batch, steps, node, dim]
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    return neg_log_p.sum() / (target.size(0) * target.size(2))


def min_ADE(preds, target, variance=cfg.scale):
    # target or preds: [batch, steps, node, dim]
    neg_log_p = ((preds - target) ** 2)
    return neg_log_p.sum() / (target.size(0) * target.size(1) * target.size(2))


def min_FDE(preds, target, variance=cfg.scale):
    # target or preds: [batch, node, dim]
    neg_log_p = ((preds - target) ** 2)
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def js_divergence(p, q, eps=1e-10):
    mid = (p + q) / 2
    return 0.5 * (kl_divergence(p, mid, eps) + kl_divergence(q, mid, eps))


def kl_gaussian(p, q, eps=1e-10):
    mu1, sigma1 = p
    mu2, sigma2 = q
    delta = (mu1 - mu2).reshape((1, -1))
    precision = torch.pinverse(sigma2)
    dim = len(mu1)
    return 0.5 * (torch.trace(precision @ sigma1) + \
                  delta.t() @ precision @ delta + \
                  torch.log(torch.det(sigma2) + eps) - \
                  torch.log(torch.det(sigma1) + eps) - dim)
