from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import model as models
import uuid
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch.utils.data as Data
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.datasets as pygeo_datasets
import torch_geometric.transforms as T

import torch_geometric.loader as dataloaders
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree
from sklearn.metrics import f1_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/home/as03347/sceneEvolution/models/gcnii",
    help="root directory",
)
parser.add_argument(
    "--expname", type=str, default="220922_pubmed_gcn_GTAdj", help="experiment name"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=5000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
parser.add_argument(
    "--wd1", type=float, default=0.01, help="weight decay (L2 loss on parameters)."
)
parser.add_argument(
    "--wd2", type=float, default=5e-4, help="weight decay (L2 loss on parameters)."
)
parser.add_argument("--layer", type=int, default=64, help="Number of layers.")
parser.add_argument("--hidden", type=int, default=64, help="hidden dimensions.")
parser.add_argument(
    "--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)."
)
parser.add_argument("--patience", type=int, default=2000, help="Patience")
parser.add_argument("--data", default="PubMed", help="dateset")
parser.add_argument(
    "--dataloader", default="GraphSAINTRandomWalkSampler", help="dateset"
)
parser.add_argument("--dev", type=int, default=0, help="device id")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha_l")
parser.add_argument("--lamda", type=float, default=0.5, help="lamda.")
parser.add_argument("--variant", type=str2bool, default=False, help="GCN* model.")
parser.add_argument(
    "--test", type=str2bool, default=True, help="evaluation on test set."
)
parser.add_argument(
    "--use_normalization",
    type=str2bool,
    default=False,
    help="use normalization constants from graphsaint",
)
parser.add_argument("--model", type=str, default="GCN", help="model name")
parser.add_argument(
    "--edge_noise_level",
    type=float,
    default=0.0042,
    help="percentage of noisy edges to add",
)
parser.add_argument(
    "--remove_interclass_edges",
    type=float,
    default=0.0,
    help="What percent of interclass edges to remove",
)
# Differentiable graph generator specific
parser.add_argument(
    "--dgm_dim",
    type=int,
    default=128,
    help="Dimensions of the linear projection layer in the DGM",
)
parser.add_argument(
    "--extra_edge_dim",
    type=int,
    default=2,
    help="extra edge dimension (for degree etc)",
)
parser.add_argument(
    "--extra_k_dim",
    type=int,
    default=1,
    help="extra k dimension (for degree etc)",
)
parser.add_argument(
    "--dgg_hard",
    type=str2bool,
    default=False,
    help="Whether to do straight through gumbel softmax"
    "(argmax in forward, softmax in backward) or just softmax top k in both",
)
parser.add_argument(
    "--dgm_temp",
    type=float,
    default=10,
    help="Gumvel softmax temperature",
)
parser.add_argument(
    "--test_noise",
    type=str2bool,
    default=False,
    help="Whether to add noise to when sampling at test time",
)
parser.add_argument(
    "--deg_mean",
    type=float,
    default=3.899,
    help="adjacecny matrix degree mean",
)
parser.add_argument(
    "--deg_std",
    type=float,
    default=5.288,
    help="adjacecny matrix degree std",
)
parser.add_argument(
    "--node_sampling_ratio",
    type=float,
    default=0.05,
    help="Sampling ratio for nodes",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=5,
    help="gradient clipping",
)
parser.add_argument(
    "--n_dgg_layers",
    type=int,
    default=1,
    help="number of dgg layers",
)
parser.add_argument(
    "--pre_normalize_adj",
    type=str2bool,
    default=False,
    help="pre normalize adjacency matrix outside network",
)
parser.add_argument(
    "--dgg_adj_input",
    type=str,
    default="input_adj",
    help="type of adjacency matrix to use for DGG, input_adj refers to the "
    "original input adjacency matrix, anything else refers to using the "
    "learned adjancency matrix",
)
parser.add_argument(
    "--dgg_mode_edge_net",
    type=str,
    default="u-v-dist",
    choices=["u-v-dist", "u-v-A_uv", "u-v-deg", "edge_conv", "A_uv"],
    help="mode for the edge_prob_net in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--dgg_mode_k_net",
    type=str,
    default="pass",
    choices=["pass", "input_deg", "gcn-x-deg", "x"],
    help="mode for the k_estimate_net in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--dgg_mode_k_select",
    type=str,
    default="edge_p-cdf",
    choices=["edge_p-cdf", "k_only", "k_times_edge_prob"],
    help="mode for the k_selector in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--graphsaint_bs",
    type=int,
    default=2000,
    help="batch size of sampled subgraph using graphsaint",
)
parser.add_argument(
    "--graphsaint_wl",
    type=int,
    default=2,
    help="walk length of sampled subgraph using graphsaint",
)


def save_checkpoint(fn, args, epoch, model, optimizer, lr_scheduler):
    torch.save(
        {
            "args": args.__dict__,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else 'null',
        },
        fn,
    )


def train(args, model, optimizer, loader, device, epoch, writer):
    model.train()
    total_loss = total_examples = total_acc = 0
    for data in loader:
        # parse data
        data = data.to(device)
        batch_adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        if args.edge_noise_level > 0.0:
            batch_adj = add_noisy_edges(batch_adj, noise_level=args.edge_noise_level)
        batch_adj = sparse_mx_to_torch_sparse_tensor(batch_adj).to(device)
        batch_feature = data.x.to(device)
        batch_label = data.y.to(device)

        if args.remove_interclass_edges > 0:
            # remove_intercommunity_edges(data, adj)
            batch_adj = remove_interclass_edges(batch_adj, batch_label)

        # zero grads
        optimizer.zero_grad()

        # forward pass
        output, out_adj, x_dgg = model(batch_feature, batch_adj, epoch, writer)
        loss = F.nll_loss(output[data.train_mask], batch_label[data.train_mask])
        acc_train = accuracy(output[data.train_mask], batch_label[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_acc += acc_train.item() * data.num_nodes
        total_examples += data.num_nodes
    loss_train = total_loss / total_examples
    acc_train = total_acc / total_examples
    return loss_train, acc_train


@torch.no_grad()
def validate(args, model, loader, device, epoch, writer):
    model.eval()

    total_test_acc = total_examples = 0
    for data in loader:
        data = data.to(device)
        batch_adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        if args.edge_noise_level > 0.0:
            batch_adj = add_noisy_edges(batch_adj, noise_level=args.edge_noise_level)

        batch_adj = sparse_mx_to_torch_sparse_tensor(batch_adj).to(device)
        batch_feature = data.x.to(device)
        batch_label = data.y.to(device)

        if args.remove_interclass_edges > 0:
            batch_adj = remove_interclass_edges(batch_adj, batch_label)

        output, out_adj, x_dgg = model(batch_feature, batch_adj, epoch, writer)
        pred = output.argmax(dim=-1)
        correct = pred.eq(batch_label)

        total_test_acc += (
            correct[data["val_mask"]].sum().item()
            / data["val_mask"].sum().item()
            * data.num_nodes
        )
        total_examples += data.num_nodes

    test_acc = total_test_acc / total_examples
    return None, test_acc


@torch.no_grad()
def test(args, model, loader, device, epoch, writer):
    model.eval()

    total_test_acc = total_examples = 0
    for data in loader:
        data = data.to(device)
        batch_adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        if args.edge_noise_level > 0.0:
            batch_adj = add_noisy_edges(batch_adj, noise_level=args.edge_noise_level)

        batch_adj = sparse_mx_to_torch_sparse_tensor(batch_adj).to(device)
        batch_feature = data.x.to(device)
        batch_label = data.y.to(device)

        if args.remove_interclass_edges > 0:
            batch_adj = remove_interclass_edges(batch_adj, batch_label)

        output, out_adj, x_dgg = model(batch_feature, batch_adj, epoch, writer)
        pred = output.argmax(dim=-1)
        correct = pred.eq(data.y.to(device))

        total_test_acc += (
            correct[data["test_mask"]].sum().item()
            / data["test_mask"].sum().item()
            * data.num_nodes
        )
        total_examples += data.num_nodes

    test_acc = total_test_acc / total_examples
    return None, test_acc


def test_best(
    args, model, test_feat, test_adj, test_labels, test_nodes, loss_fcn, device
):
    model.load_state_dict(torch.load(checkpt_file)["model_state_dict"])
    loss_test = 0
    acc_test = 0
    for batch in range(2):
        batch_adj = test_adj[batch].to(device)
        batch_feature = test_feat[batch].to(device)
        batch_label = test_labels[batch].to(device)
        score, loss = evaluate(
            batch_feature, model, test_nodes[batch], batch_adj, batch_label, loss_fcn
        )
        loss_test += loss
        acc_test += score
    acc_test /= 2
    loss_test /= 2
    return loss_test, acc_test


# adapted from DGL
def evaluate(feats, model, mask, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        output = model(feats, subgraph)
        loss_data = loss_fcn(output[mask], labels[mask].float())
        predict = np.where(output[mask].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels[mask].data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def load_data(args):
    if "DGG" not in args.model:
        args.pre_normalize_adj = False

    root = "/home/as03347/sceneEvolution/data/graph_data/{}".format(args.data)
    dataset = pygeo_datasets.__dict__["Planetoid"](
        root=root, name=args.data, split="public", transform=T.NormalizeFeatures()
    )
    data = dataset[0]

    train_input_nodes = data.train_mask
    test_input_nodes = data.test_mask

    train_loader = dataloaders.NeighborLoader(
        data, [50] * 2, shuffle=True, input_nodes=train_input_nodes
    )
    test_loader = dataloaders.NeighborLoader(
        data, [3] * 2, shuffle=True, input_nodes=test_input_nodes
    )

    # cluster_data = dataloaders.ClusterData(data, num_parts=200, recursive=False,
    #                            save_dir=dataset.processed_dir)
    # loader = dataloaders.ClusterLoader(cluster_data, batch_size=20, shuffle=True,
    #                              num_workers=4)
    return dataset, train_loader, test_loader


if __name__ == "__main__":

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialise directories
    outdir = os.path.join(args.root, "outputs")
    expdir = os.path.join(outdir, args.expname)
    tbdir = os.path.join(expdir, "tb")
    codedir = os.path.join(expdir, "code")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    os.makedirs(codedir, exist_ok=True)
    checkpt_file = os.path.join(expdir, uuid.uuid4().hex + ".pt")
    print(checkpt_file)

    # Make copy of code
    python_files = [f for f in os.listdir(args.root) if ".py" in f]
    for f in python_files:
        shutil.copyfile(src=os.path.join(args.root, f), dst=os.path.join(codedir, f))

    # Tensorboard writer
    writer = SummaryWriter(tbdir)

    # Load data
    dataset, train_loader, test_loader = load_data(args)
    cudaid = "cuda"
    device = torch.device(cudaid)

    # Load model
    model = models.__dict__[args.model](
        nfeat=dataset.num_features,
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=dataset.num_classes,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=args.variant,
        args=args,
    ).to(device)

    if "GCNII" in args.model:
        optimizer = optim.Adam(
            [
                {"params": model.params1, "weight_decay": args.wd1},
                {"params": model.params2, "weight_decay": args.wd2},
            ],
            lr=args.lr,
        )
    else:
        optimizer = optim.Adam(
            [
                dict(params=model.params1, weight_decay=5e-4),
                dict(params=model.params2, weight_decay=0),
            ],
            lr=args.lr,
        )  # Only perform weight-decay on first convolution.

    # Run
    t_total = time.time()
    bad_counter = 0
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_train, acc_train = train(
            args, model, optimizer, train_loader, device, epoch, writer
        )
        acc_test = test(args, model, test_loader, device, epoch, writer)[1]

        if (epoch + 1) % 1 == 0:
            print(
                "Epoch:{:04d}".format(epoch + 1),
                "train",
                "loss:{:.3f}".format(loss_train),
                "acc:{:.3f}".format(acc_train),
                "| test",
                "f1:{:.3f}".format(acc_test * 100),
            )

            writer.add_scalar("train/loss", loss_train, epoch)
            writer.add_scalar("train/acc", acc_train, epoch)
            writer.add_scalar("test/acc", acc_test, epoch)

    if args.test:
        # acc = test_best(
        #     args, model, test_feat, test_adj, test_labels,
        #     test_nodes, loss_fcn, device
        # )[1]
        acc = acc_test

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print("Load {}th epoch".format(best_epoch))
    print("Test" if args.test else "Val", "f1.:{:.2f}".format(acc * 100))
