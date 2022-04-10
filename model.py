import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgm import DGG_LearnableK
from utils import torch_normalized_adjacency
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNII_DGG(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args):
        super(GCNII_DGG, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        self.dgm_dim = args.dgm_dim
        self.st_gumbel_softmax = args.st_gumbel_softmax
        self.self_loops_noise = args.self_loops_noise
        self.k_bias = args.k_bias
        self.dgg_dist_fn = args.dgg_dist_fn
        self.k_net_input = args.k_net_input
        self.hs_start = args.hs_start
        self.hs_end = args.hs_end
        self.dgm_temp = args.dgm_temp

        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK(
                    in_dim=nhidden, latent_dim=nhidden, k_bias=self.k_bias,
                    hard=self.st_gumbel_softmax, self_loops_noise=self.self_loops_noise,
                    dist_fn=self.dgg_dist_fn, k_net_input=args.k_net_input,
                    hs_start=self.hs_start, hs_end=self.hs_end,
                )
            )

        self.params1 = list(self.convs.parameters())
        self.params1.extend(list(self.dggs.parameters()))
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj_og):
        """
        x: [N, dim]
        adj: [N, N]
        """

        # x = x[:5]
        # adj = adj.to_dense()[:5].to_sparse()
        # print(adj.to_dense().sum(-1).mean(), adj.to_dense().sum(-1).max(), adj.to_dense().sum(-1).min())

        _layers = []

        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))

        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):

            # learn structure and get adjacency matrix
            if i < len(self.dggs):
                adj = self.dggs[i](
                    x=layer_inner.unsqueeze(0), temp=self.dgm_temp, noise=self.training
                )
                adj = adj.squeeze(0)    # [N, N]
                adj = torch_normalized_adjacency(adj) * adj_og.to_dense()
                # print('adj', adj.sum(-1).mean().item(), adj.sum(-1).max().item(), adj.sum(-1).mean().item())

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return F.log_softmax(layer_inner, dim=1)


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(
                GraphConvolution(nhidden, nhidden, variant=variant, residual=True)
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == "__main__":
    pass
