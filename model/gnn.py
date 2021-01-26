import torch
from torch import nn
from torch_geometric.utils import scatter_
from torch_geometric.utils import softmax as gsoftmax
from torch.autograd import Variable
import config as cfg
import math


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x, es, f_e=None, agg='mean'):
        msg, idx, size = self.message(x, es, f_e)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg, idx, size, agg='mean'):
        # Only 3 types of aggregation are supported.
        # msg: [E, ..., dim * 2]
        # idx: [E]
        assert agg in {'add', 'mean', 'max'}
        return scatter_(agg, msg, idx, dim_size=size)

    def node2edge(self, x_i, x_o, f_e):
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x, es, f_e=None, option='o2i'):
        # x: [size, ..., dim]
        # es: [2, E]
        if option == 'i2o':
            row, col = es
        if option == 'o2i':
            col, row = es

        x_i, x_o = x[row], x[col]
        msg = self.node2edge(x_i, x_o, f_e)
        return msg, col, len(x)

    def update(self, x):
        return x


class MyGAT(GNN):
    def __init__(self, att, dim_in, dim_out=None):
        super(MyGAT, self).__init__()
        self.att = att
        if dim_out is None:
            dim_out = dim_in
        self.lin = nn.Linear(dim_in, dim_out)
        self.e2n = nn.Linear(dim_out * 2, dim_out)

    def propagate(self, x, es, agg='add'):
        msg, idx, size = self.message(x, es)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg, idx, size, agg='add'):
        # Only 3 types of aggregation are supported.
        # msg: [E, ..., head, dim * 2]
        # idx: [E]
        assert agg in {'add', 'mean', 'max'}
        x = scatter_(agg, msg, idx, dim_size=size)
        # mean over multi-head attention
        x = x.mean(-2)
        return x

    def message(self, x, es):
        # x: [size, ..., dim]
        # es: [2, E]
        row, col = es
        x_i, x_o = x[row], x[col]
        # att: [E, ..., head]
        att = self.att(x_i, x_o)
        att = gsoftmax(att, col, x.shape[0])
        # msg: [E, ..., dim * 2]
        # msg = torch.cat([x_i, x_o], dim=-1)
        # msg = x_o
        msg = torch.cat([x_i, x_o], dim=-1)
        msg = self.e2n(msg)
        # reshape
        att = att.unsqueeze(-1)
        msg = msg.unsqueeze(-2)
        msg = msg * att
        return msg, col, len(x)

    def forward(self, x, es):
        # x: [size, ..., dim]
        x = self.lin(x)
        x = self.propagate(x, es)
        return x


class GAT(GNN):
    def __init__(self, att, dim_in, dim_out=None):
        super(GAT, self).__init__()
        self.att = att
        if dim_out is None:
            dim_out = dim_in

    def propagate(self, x, es, agg='add'):
        msg, idx, size = self.message(x, es)
        x = self.aggregate(msg, idx, size, agg)
        return x

    def aggregate(self, msg, idx, size, agg='add'):
        # Only 3 types of aggregation are supported.
        # msg: [E, ..., head, dim * 2]
        # idx: [E]
        assert agg in {'add', 'mean', 'max'}
        x = scatter_(agg, msg, idx, dim_size=size)
        # mean over multi-head attention
        x = x.mean(-2)
        return x

    def message(self, x, es):
        # x: [size, ..., dim]
        # es: [2, E]
        row, col = es
        x_i, x_o = x[row], x[col]
        # att: [E, ..., head]
        att = self.att(x_i, x_o)
        att = gsoftmax(att, col, x.shape[0])
        # msg: [E, ..., dim * 2]
        # msg = torch.cat([x_i, x_o], dim=-1)
        # msg = x_o
        msg = torch.cat([x_i, x_o], dim=-1)
        msg = self.e2n(msg)
        # reshape
        att = att.unsqueeze(-1)
        msg = msg.unsqueeze(-2)
        msg = msg * att
        return msg, col, len(x)

    def forward(self, x, es):
        # x: [size, ..., dim]
        x = self.lin(x)
        x = self.propagate(x, es)
        return x


class GTransformer(GNN):
    def __init__(self, dim_in, dim_out=None, do_prob=0.0):
        super(GTransformer, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.dim_out = dim_out
        self.query = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(do_prob))
        self.key = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(do_prob))
        self.value = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(do_prob))

    def propagate(self, x, es, agg='add'):
        msg, idx, size = self.message(x, es)
        x = self.aggregate(msg, idx, size, agg)
        return x

    def aggregate(self, msg, idx, size, agg='add'):
        # Only 3 types of aggregation are supported.
        # msg: [E, ..., head, dim * 2]
        # idx: [E]
        assert agg in {'add', 'mean', 'max'}
        x = scatter_(agg, msg, idx, dim_size=size)
        # mean over multi-head attention
        x = x.mean(-2)
        return x

    def message(self, x, es):
        # x: [size, ..., dim]
        # es: [2, E]
        row, col = es
        x_i, x_o = x[row], x[col]
        # att: [E, ..., head]
        query = self.query(x_i)
        key = self.key(x_o)
        alpha = (query * key).sum(-1) / math.sqrt(self.dim_out)
        att = gsoftmax(alpha, col, x.shape[0])
        # msg: [E, ..., dim * 2]
        value = self.value(x_o)
        # reshape
        att = att.unsqueeze(-1)
        value = value.unsqueeze(-2)
        value = value * att
        return value, col, len(x)

    def forward(self, x, es):
        # x: [size, ..., dim]
        x = self.propagate(x, es)
        return x


class SkipAffine(nn.Module):
    def __init__(self, gnn, n_in, n_out):
        super(SkipAffine, self).__init()
        self.gnn = gnn
        self.aff = nn.Linear(n_in, n_out)

    def forward(self, x, es, *args):
        hidden = self.gnn(x, es, *args)
        aff = self.aff(x)
        return hidden + aff


class mixprop(nn.Module):
    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid, gdep=1,do_prob=0,alpha=0.05):
        super(mixprop, self).__init__()
        #self.nconv = nconv()
        self.mlp = nn.Sequential(nn.Linear((gdep+1)*msg_hid,msg_out), nn.ReLU(),)
        self.gdep = gdep
        #self.dropout = dropout
        self.alpha = alpha
        self.msg_out = msg_out
        self.msgs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * n_in_node, msg_hid),
                    nn.ReLU(),
                    nn.Dropout(do_prob),
                    #nn.Linear(msg_hid, msg_out),
                    #nn.ReLU(),
                )
                for _ in range(edge_types)
            ])


    def forward(self,x,adj,idx,norm):
        msgs = Variable(torch.zeros(x.size(0), x.size(1), x.size(2), self.msg_out))
        if cfg.gpu:
            msgs = msgs.cuda()
        for i in range(idx, len(self.msgs)):
            msg = self.msgs[i](x)
            h = msg * torch.select(adj, -1, i).unsqueeze(-1)
            out = [h]
            for j in range(self.gdep):
                h = self.alpha * h + (1 - self.alpha) * h * torch.select(adj, -1, i).unsqueeze(-1)
                out.append(h)
            out = torch.cat(out, dim=-1)
            out = self.mlp(out)
            out = out / norm
            msgs += out
        return msgs
