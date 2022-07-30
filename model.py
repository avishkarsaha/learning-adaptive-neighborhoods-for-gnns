import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgm import DGG_LearnableK, DGG_LearnableK_debug
from utils import torch_normalized_adjacency
from torch_geometric.datasets import KarateClub


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


class DenseGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(DenseGraphConvolution, self).__init__()
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
        hi = torch.mm(adj, input)
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


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, A=None, cached=False):
        super(GCNConv, self).__init__()
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))

    def forward(self, x, adj):
        """
        Args:
            x:
            adj: normalized adjacency matrix

        Returns:

        """
        Ax = torch.mm(adj, x)
        # print('Ax mu: {:.3f} std: {:.3f}'.format(Ax.mean().item(), Ax.std().item()),)
        AxW = torch.mm(Ax, self.W)
        # print('AxW mu: {:.3f} std: {:.3f}'.format(AxW.mean().item(), AxW.std().item()), )
        out = torch.relu(AxW)
        return out


class GCNII(nn.Module):
    def __init__(
        self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args
    ):
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

    def normalize_adj(self, A):
        # assert no self loops
        assert A[torch.arange(len(A)), torch.arange(len(A))].sum() == 0

        # add self loops
        A_hat = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, adj, epoch=None, writer=None):
        adj = adj.to_dense()
        adj = self.normalize_adj(adj)

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
    def __init__(
        self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args
    ):
        super(GCNII_DGG, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(DenseGraphConvolution(nhidden, nhidden, variant=variant))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        self.dgg_adj_input = args.dgg_adj_input
        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK_debug(in_dim=nfeat, latent_dim=nhidden, args=args)
            )

        self.params1 = list(self.convs.parameters())
        self.params1.extend(list(self.dggs.parameters()))
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def normalize_adj(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        row_sum = A_hat.sum(-1)
        row_sum = (row_sum) ** -0.5
        D = torch.diag(row_sum)
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, in_adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        # add self-loops
        in_adj = (
            in_adj.to_dense() + torch.eye(in_adj.shape[0], device=in_adj.device)
        ).to_sparse()

        # coalesce to track grads
        unnorm_adj = in_adj.coalesce()
        for i, con in enumerate(self.convs):
            if i < len(self.dggs):
                if self.dgg_adj_input == "input_adj":
                    # always use input adjacency
                    unnorm_adj = self.dgg_net(x, i, in_adj.coalesce(), writer, epoch)
                else:
                    # use updated adjacency
                    unnorm_adj = self.dgg_net(x, i, unnorm_adj, writer, epoch)
                norm_adj = self.normalize_adj(unnorm_adj.to_dense())

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, norm_adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return F.log_softmax(layer_inner, dim=1)

    def dgg_net(self, x, i, unnorm_adj, writer, epoch):
        # learn adjacency (sparse tensor)
        adj = self.dggs[i](
            x=x, in_adj=unnorm_adj, noise=self.training, writer=writer, epoch=epoch
        )
        return adj


class GCNII_DGG_viz(nn.Module):
    def __init__(
        self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args
    ):
        super(GCNII_DGG_viz, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(DenseGraphConvolution(nhidden, nhidden, variant=variant))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))

        self.dgm_dim = args.dgm_dim
        self.st_gumbel_softmax = args.st_gumbel_softmax
        self.self_loops_noise = args.self_loops_noise
        self.k_bias = args.k_bias
        self.dgg_dist_fn = args.dgg_dist_fn
        self.k_net_input = args.k_net_input
        self.deg_mean = args.deg_mean
        self.deg_std = args.deg_std
        self.dgm_temp = args.dgm_temp

        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK(
                    in_dim=nhidden,
                    latent_dim=nhidden,
                    k_bias=self.k_bias,
                    hard=self.st_gumbel_softmax,
                    self_loops_noise=self.self_loops_noise,
                    dist_fn=self.dgg_dist_fn,
                    k_net_input=args.k_net_input,
                    degree_mean=self.deg_mean,
                    degree_std=self.deg_std,
                )
            )

        self.params1 = list(self.convs.parameters())
        self.params1.extend(list(self.dggs.parameters()))
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj, epoch=None, writer=None):
        """
        x: input features [N, dim]
        adj: dense adjacency matrix [N, N]
        """
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):

            # learn structure and get adjacency matrix
            if i < len(self.dggs):
                adj = self.dggs[i](
                    x=layer_inner,
                    in_adj=adj,
                    temp=self.dgm_temp,
                    noise=False,
                    writer=writer,
                    epoch=epoch,
                )
                adj = adj.squeeze(0)
                # if writer is not None:
                #     writer.add_histogram(
                #         'train/our_node_degree',
                #         (adj > 0.5).float().sum(-1), epoch
                #     )
                #     writer.add_histogram('train/k', k.flatten(), epoch)

                adj = torch_normalized_adjacency(adj, mode="self_loops_present")
                # adj = adj.to_sparse()
                # print('adj', adj.sum(-1).mean().item(), adj.sum(-1).max().item(), adj.sum(-1).mean().item())

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return F.log_softmax(layer_inner, dim=1)


class GCNIIppi(nn.Module):
    def __init__(
        self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args
    ):
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

    def forward(self, x, adj, writer=None, epoch=None):
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


class GCNIIppi_DGG(nn.Module):
    def __init__(
        self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, args
    ):
        super(GCNIIppi_DGG, self).__init__()
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

        self.dgg_adj_input = args.dgg_adj_input
        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK_debug(in_dim=nfeat, latent_dim=nhidden, args=args)
            )

    def normalize_adj(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        row_sum = A_hat.sum(-1)
        row_sum = (row_sum) ** -0.5
        D = torch.diag(row_sum)
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, in_adj, writer=None, epoch=None):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        # add self-loops
        in_adj = (
            in_adj.to_dense() + torch.eye(in_adj.shape[0], device=in_adj.device)
        ).to_sparse()

        # coalesce to track grads
        unnorm_adj = in_adj.coalesce()
        for i, con in enumerate(self.convs):
            if i < len(self.dggs):
                if self.dgg_adj_input == "input_adj":
                    # always use input adjacency
                    unnorm_adj = self.dgg_net(x, i, in_adj.coalesce(), writer, epoch)
                else:
                    # use updated adjacency
                    unnorm_adj = self.dgg_net(x, i, unnorm_adj, writer, epoch)
                norm_adj = self.normalize_adj(unnorm_adj.to_dense())

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, norm_adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner

    def dgg_net(self, x, i, unnorm_adj, writer, epoch):
        # learn adjacency (sparse tensor)
        adj = self.dggs[i](
            x=x, in_adj=unnorm_adj, noise=self.training, writer=writer, epoch=epoch
        )
        return adj


class GCN(torch.nn.Module):
    """
    GCN for binary node classification
    """
    def __init__(self, nfeat=32, nlayers=None, nhidden=32, nclass=10, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())

    def normalize_adj(self, A):
        # assert no self loops
        if A[torch.arange(len(A)), torch.arange(len(A))].sum() != 0:
            A[torch.arange(len(A)), torch.arange(len(A))] = 0

        # add self loops
        A_hat = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        adj = adj.to_dense()
        adj = self.normalize_adj(adj)

        x = F.dropout(self.conv1(x, adj), training=self.training)
        if writer is not None:
            writer.add_histogram("gcn_conv1_dist", x, epoch)
            # print(
            #     "conv1 mu: {:.5f} std: {:.5f}".format(x.mean().item(), x.std().item())
            # )

        x = self.conv2(x, adj)
        if writer is not None:
            writer.add_histogram("gcn_conv2_dist", x, epoch)
            # print(
            #     'conv2 mu: {:.5f} std: {:.5f}'.format(x.mean().item(), x.std().item())
            # )

        out = F.log_softmax(x, dim=-1)
        return out


class GCN_MultiClass(torch.nn.Module):
    """
    GCN for multi-label node classification
    """
    def __init__(self, nfeat=32, nlayers=None, nhidden=32, nclass=10, **kwargs):
        super(GCN_MultiClass, self).__init__()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())

    def normalize_adj(self, A):
        # assert no self loops
        if A[torch.arange(len(A)), torch.arange(len(A))].sum() != 0:
            A[torch.arange(len(A)), torch.arange(len(A))] = 0

        # add self loops
        A_hat = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        adj = adj.to_dense()
        adj = self.normalize_adj(adj)

        x = F.dropout(self.conv1(x, adj), training=self.training)
        if writer is not None:
            writer.add_histogram("gcn_conv1_dist", x, epoch)
            # print(
            #     "conv1 mu: {:.5f} std: {:.5f}".format(x.mean().item(), x.std().item())
            # )

        x = self.conv2(x, adj)
        if writer is not None:
            writer.add_histogram("gcn_conv2_dist", x, epoch)
            # print(
            #     'conv2 mu: {:.5f} std: {:.5f}'.format(x.mean().item(), x.std().item())
            # )

        out = x.sigmoid()
        return out


class GCN_LargeGraphs(torch.nn.Module):
    def __init__(self, nfeat=32, nlayers=None, nhidden=32, nclass=10, **kwargs):
        super(GCN_LargeGraphs, self).__init__()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())

    def normalize_adj(self, A):
        # assert no self loops
        assert A[torch.arange(len(A)), torch.arange(len(A))].sum() == 0

        # add self loops
        A_hat = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        adj = adj.to_dense()
        adj = self.normalize_adj(adj)

        x = F.dropout(self.conv1(x, adj), training=self.training)
        if writer is not None:
            writer.add_histogram("gcn_conv1_dist", x, epoch)
            print(
                "conv1 mu: {:.5f} std: {:.5f}".format(x.mean().item(), x.std().item())
            )

        x = self.conv2(x, adj)
        if writer is not None:
            writer.add_histogram("gcn_conv2_dist", x, epoch)
            # print(
            #     'conv2 mu: {:.5f} std: {:.5f}'.format(x.mean().item(), x.std().item())
            # )

        out = nn.Sigmoid(x)
        return out


class GCN_debug(torch.nn.Module):
    def __init__(self, nfeat=32, nlayers=None, nhidden=32, nclass=10, **kwargs):
        super(GCN_debug, self).__init__()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())

    def normalize_adj(self, A):
        A_hat = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        adj = adj.to_dense()
        adj = self.normalize_adj(adj)

        if writer is not None:
            u = x[torch.where(adj > 0)[0]]
            v = x[torch.where(adj > 0)[1]]
            dist = torch.linalg.vector_norm(u - v, dim=-1, ord=2)

        #     deg_diff = torch.abs(out_adj.sum(-1) - in_adj.sum(-1))
        #     writer.add_scalar('values/deg_diff_std', deg_diff.std(), epoch)
        #     writer.add_scalar('values/deg_diff_mean', deg_diff.mean(), epoch)
        #     writer.add_scalar('values/deg_std', out_adj.sum(-1).std(), epoch)
        #     writer.add_scalar('values/deg_mean', out_adj.sum(-1).mean(), epoch)

        x = F.dropout(self.conv1(x, adj), training=self.training)
        if epoch % 10 == 0:
            print(
                "conv1 mu: {:.5f} std: {:.5f}".format(x.mean().item(), x.std().item())
            )
        x = self.conv2(x, adj)
        if epoch % 10 == 0:
            print(
                "conv2 mu: {:.5f} std: {:.5f}".format(x.mean().item(), x.std().item())
            )
        return x, adj


class GCN_DGG(torch.nn.Module):
    def __init__(
        self, nfeat=32, nlayers=None, nhidden=32, nclass=10, args=None, **kwargs
    ):
        super(GCN_DGG, self).__init__()

        # GCN layers
        self.convs = nn.ModuleList()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)
        self.convs.append(self.conv1)
        self.convs.append(self.conv2)

        self.dgg_adj_input = args.dgg_adj_input
        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK_debug(in_dim=nfeat, latent_dim=nhidden, args=args)
            )

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())
        self.params2.extend(list(self.dggs.parameters()))

    def normalize_adj(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        row_sum = A_hat.sum(-1)
        row_sum = (row_sum) ** -0.5
        D = torch.diag(row_sum)
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def normalize_adj_gcn(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, in_adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        # print('n edges before self loops', in_adj.to_dense().sum())
        # add self-loops
        in_adj = (
            in_adj.to_dense() + torch.eye(in_adj.shape[0], device=in_adj.device)
        ).to_sparse()

        if epoch == 0:
            diagonal_w = in_adj.to_dense()[
                torch.arange(in_adj.shape[0]), torch.arange(in_adj.shape[0])
            ] / in_adj.to_dense().sum(-1)

            print(
                "in diag w {:.5f} {:.5f}".format(
                    diagonal_w.mean().item(), diagonal_w.std().item()
                )
            )
        # coalesce to track grads
        unnorm_adj = in_adj.coalesce()

        for i, conv in enumerate(self.convs):
            if i < len(self.dggs):
                if self.dgg_adj_input == "input_adj":
                    # always use input adjacency
                    unnorm_adj = self.dgg_net(x, i, in_adj.coalesce(), writer, epoch)
                else:
                    # use updated adjacency
                    unnorm_adj = self.dgg_net(x, i, unnorm_adj, writer, epoch)
                norm_adj = self.normalize_adj(unnorm_adj.to_dense())

            diagonal_w = norm_adj[
                torch.arange(norm_adj.shape[0]), torch.arange(norm_adj.shape[0])
            ] / norm_adj.sum(-1)
            # print('out diag w {:.5f} {:.5f}'.format(
            #     diagonal_w.mean().item(), diagonal_w.std().item()))

            x = conv(x, norm_adj)

            if i < len(self.convs) - 1:
                x = F.dropout(x, training=self.training)

            if writer is not None:
                writer.add_histogram("gcn_conv{}_dist".format(i + 1), x, epoch)
                # print(
                #     'conv{} mu: {:.5f} std: {:.5f}'.format(i + 1, x.mean().item(),
                #                                           x.std().item())
                # )

        out = F.log_softmax(x, dim=-1)

        # in_adj = in_adj.to_dense()
        # if writer is not None:
        #     deg_diff = torch.abs(out_adj.sum(-1) - in_adj.sum(-1))
        #     writer.add_scalar('values/deg_diff_std', deg_diff.std(), epoch)
        #     writer.add_scalar('values/deg_diff_mean', deg_diff.mean(), epoch)
        #     writer.add_scalar('values/deg_std', out_adj.sum(-1).std(), epoch)
        #     writer.add_scalar('values/deg_mean', out_adj.sum(-1).mean(), epoch)

        return out

    def dgg_net(self, x, i, unnorm_adj, writer, epoch):
        # learn adjacency (sparse tensor)
        adj = self.dggs[i](
            x=x, in_adj=unnorm_adj, noise=self.training, writer=writer, epoch=epoch
        )
        return adj


class GCN_DGG_LargeGraphs(torch.nn.Module):
    def __init__(
        self, nfeat=32, nlayers=None, nhidden=32, nclass=10, args=None, **kwargs
    ):
        super(GCN_DGG, self).__init__()

        # GCN layers
        self.convs = nn.ModuleList()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)
        self.convs.append(self.conv1)
        self.convs.append(self.conv2)

        self.dgg_adj_input = args.dgg_adj_input
        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK_debug(in_dim=nfeat, latent_dim=nhidden, args=args)
            )

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())
        self.params2.extend(list(self.dggs.parameters()))

    def normalize_adj(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        row_sum = A_hat.sum(-1)
        row_sum = (row_sum) ** -0.5
        D = torch.diag(row_sum)
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def normalize_adj_gcn(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, in_adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        # print('n edges before self loops', in_adj.to_dense().sum())
        # add self-loops
        in_adj = (
            in_adj.to_dense() + torch.eye(in_adj.shape[0], device=in_adj.device)
        ).to_sparse()
        # print('n edges with self loops', in_adj.to_dense().sum())
        if epoch == 0:
            diagonal_w = in_adj.to_dense()[
                torch.arange(in_adj.shape[0]), torch.arange(in_adj.shape[0])
            ] / in_adj.to_dense().sum(-1)

            print(
                "in diag w {:.5f} {:.5f}".format(
                    diagonal_w.mean().item(), diagonal_w.std().item()
                )
            )
        # coalesce to track grads
        unnorm_adj = in_adj.coalesce()

        for i, conv in enumerate(self.convs):
            if i < len(self.dggs):
                if self.dgg_adj_input == "input_adj":
                    # always use input adjacency
                    unnorm_adj = self.dgg_net(x, i, in_adj.coalesce(), writer, epoch)
                else:
                    # use updated adjacency
                    unnorm_adj = self.dgg_net(x, i, unnorm_adj, writer, epoch)
                norm_adj = self.normalize_adj(unnorm_adj.to_dense())

            diagonal_w = norm_adj[
                torch.arange(norm_adj.shape[0]), torch.arange(norm_adj.shape[0])
            ] / norm_adj.sum(-1)
            # print('out diag w {:.5f} {:.5f}'.format(
            #     diagonal_w.mean().item(), diagonal_w.std().item()))

            x = conv(x, norm_adj)

            if i < len(self.convs) - 1:
                x = F.dropout(x, training=self.training)

            if writer is not None:
                writer.add_histogram("gcn_conv{}_dist".format(i + 1), x, epoch)
                # print(
                #     'conv{} mu: {:.5f} std: {:.5f}'.format(i + 1, x.mean().item(),
                #                                           x.std().item())
                # )

        out = nn.Sigmoid(x)

        # in_adj = in_adj.to_dense()
        # if writer is not None:
        #     deg_diff = torch.abs(out_adj.sum(-1) - in_adj.sum(-1))
        #     writer.add_scalar('values/deg_diff_std', deg_diff.std(), epoch)
        #     writer.add_scalar('values/deg_diff_mean', deg_diff.mean(), epoch)
        #     writer.add_scalar('values/deg_std', out_adj.sum(-1).std(), epoch)
        #     writer.add_scalar('values/deg_mean', out_adj.sum(-1).mean(), epoch)

        return out

    def dgg_net(self, x, i, unnorm_adj, writer, epoch):
        # learn adjacency (sparse tensor)
        adj = self.dggs[i](
            x=x, in_adj=unnorm_adj, noise=self.training, writer=writer, epoch=epoch
        )
        return adj


class GCN_DGG_debug(torch.nn.Module):
    def __init__(
        self, nfeat=32, nlayers=None, nhidden=32, nclass=10, args=None, **kwargs
    ):
        super(GCN_DGG_debug, self).__init__()

        # GCN layers
        self.convs = nn.ModuleList()
        self.conv1 = GCNConv(nfeat, nhidden)
        self.conv2 = GCNConv(nhidden, nclass)
        self.convs.append(self.conv1)
        self.convs.append(self.conv2)

        self.dgg_adj_input = args.dgg_adj_input
        self.dggs = nn.ModuleList()
        for _ in range(args.n_dgg_layers):
            self.dggs.append(
                DGG_LearnableK_debug(in_dim=nfeat, latent_dim=nhidden, args=args)
            )

        self.params1 = list(self.conv1.parameters())
        self.params2 = list(self.conv2.parameters())
        self.params2.extend(list(self.dggs.parameters()))

    def normalize_adj(self, A_hat):
        """
        renormalisation of adjacency matrix
        Args:
            A_hat: adj mat with self loops [N, N]

        Returns:
            A_hat: renormalized adjaceny [N, N]

        """
        row_sum = A_hat.sum(-1)
        row_sum = (row_sum) ** -0.5
        D = torch.diag(row_sum)
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        return A_hat

    def forward(self, x, in_adj, epoch=None, writer=None):
        """
        Args:
            x: node features
            A: sparse unnormalized adjacency matrix without self loops
            epoch: epoch number
            writer: tensorboard summary writer

        Returns:
            out: class predictions for each node
        """
        # add self-loops
        in_adj = (
            in_adj.to_dense() + torch.eye(in_adj.shape[0], device=in_adj.device)
        ).to_sparse()

        # coalesce to track grads
        unnorm_adj = in_adj.coalesce()

        for i, conv in enumerate(self.convs):
            if i < len(self.dggs):
                if self.dgg_adj_input == "input_adj":
                    # always use input adjacency
                    unnorm_adj, debug_dict = self.dgg_net(
                        x, i, in_adj.coalesce(), writer, epoch
                    )
                else:
                    # use updated adjacency
                    unnorm_adj, debug_dict = self.dgg_net(
                        x, i, unnorm_adj, writer, epoch
                    )

                # convert to dense tensor and normal
                unnorm_adj = unnorm_adj.to_dense()
                # if epoch % 1000 == 0:
                #     print(
                #         'unnorm adj deg mu: {:.5f} std: {:.5f}'.format(
                #             unnorm_adj.sum(-1).mean().item(),
                #             unnorm_adj.sum(-1).std().item()
                #         ))
                norm_adj = self.normalize_adj(unnorm_adj)

            x = conv(x, norm_adj)

            if i < len(self.convs) - 1:
                x = F.dropout(x, training=self.training)

            # if epoch % 1000 == 0:
            #     print(
            #         'conv{} mu: {:.5f} std: {:.5f}'.format(
            #             i + 1, x.mean().item(), x.std().item()
            #         )
            #     )
        # return x, debug_dict
        return x

    def dgg_net(self, x, i, unnorm_adj, writer, epoch):
        # learn adjacency (sparse tensor)
        adj = self.dggs[i](
            x=x, in_adj=unnorm_adj, noise=self.training, writer=writer, epoch=epoch
        )
        return adj


if __name__ == "__main__":

    pass
