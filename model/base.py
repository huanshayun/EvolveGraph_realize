from torch import nn
import torch
import torch.nn.functional as F
from utils.general import prod
from utils.torch_extension import my_bn
from torch.nn.modules import BatchNorm1d


class MyMLP(nn.Module):
    '''2-layer MLP'''
    def __init__(self, dim_in, dim_out,
                 dim_hid=None, act=None):
        super(MyMLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            act,
            nn.Linear(dim_hid, dim_out)
        )

    def forward(self, x):
        return self.model(x)


class Lin(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_out, do_prob=0.):
        super(Lin, self).__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(prod(inputs.shape[:-1]), -1)
        x = self.bn(x)
        return x.view(*inputs.shape[:-1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        return self.batch_norm(x)


class XMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(XMLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        # self.bn = nn.BatchNorm1d(n_out)
        # added
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.dropout_prob = do_prob

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(prod(inputs.shape[:-1]), -1)
        x = self.bn(x)
        return x.view(*inputs.shape[:-1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        # added
        x = my_bn(x, self.bn2)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return x


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        # added
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.dropout_prob = do_prob

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(prod(inputs.shape[:-1]), -1)
        x = self.bn(x)
        return x.view(*inputs.shape[:-1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        # added
        x = my_bn(x, self.bn2)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

'''
class TripleMLP(nn.Module):
    """Threee-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(TripleMLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        self.bn1 = nn.BatchNorm1d(n_hid)
        # added
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.bn3 = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs, bn):
        x = inputs.view(prod(inputs.shape[:-1]), -1)
        x = bn(x)
        return x.view(*inputs.shape[:-1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        # added
        x = self.batch_norm(x, self.bn1)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        x = F.elu(self.fc2(x))
        # added
        x = self.batch_norm(x, self.bn2)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        x = F.elu(self.fc3(x))
        # added
        x = self.batch_norm(x, self.bn3)
        #x = F.dropout(x, self.dropout_prob, training=self.training)
        return x
'''

class MyBN1d(nn.Module):
    def __init__(self, n_in, dim=-1):
        super(MyBN1d, self).__init__()
        self.bn = BatchNorm1d(n_in)
        self.dim = dim

    def init_weights(self):
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero()

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(self.dim, -1).contiguous()
            h = my_bn(x, self.bn)
            h = h.transpose(self.dim, -1).contiguous()
        else:
            h = my_bn(x, self.bn)
        return h


class CirConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel):
        super(CirConv1d, self).__init__()
        self.kernel = kernel
        self.model = nn.Conv1d(dim_in, dim_out, kernel_size=kernel)

    def forward(self, x):
        x = F.pad(x, (0, self.kernel-1), mode='circular')
        return self.model(x)


class MultiConv1d(nn.Module):
    def __init__(self, dim_in, dim_outs, kernels):
        super(MultiConv1d, self).__init__()
        self.models = nn.ModuleList([
            CirConv1d(dim_in, dim_out, kernel)
            for dim_out, kernel in zip(dim_outs, kernels)
        ])

    def forward(self, x):
        h = [model(x) for model in self.models]
        h = torch.cat(h, dim=1)
        return h
