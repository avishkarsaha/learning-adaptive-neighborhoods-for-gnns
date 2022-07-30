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
import torch_geometric.transforms as T
import shutil

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/home/as03347/sceneEvolution/models/gcnii",
    help="root directory",
)
parser.add_argument(
    "--expname",
    type=str,
    default="220730_cora_gcndgg_debug",
    help="experiment name",
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
parser.add_argument("--layer", type=int, default=16, help="Number of layers.")
parser.add_argument("--hidden", type=int, default=64, help="hidden dimensions.")
parser.add_argument(
    "--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)."
)
parser.add_argument("--patience", type=int, default=2000, help="Patience")
parser.add_argument("--data", default="cora", help="dateset")
parser.add_argument("--split", default="public", help="dateset")
parser.add_argument("--dev", type=int, default=0, help="device id")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha_l")
parser.add_argument("--lamda", type=float, default=0.5, help="lamda.")
parser.add_argument("--variant", type=str2bool, default=False, help="GCN* model.")
parser.add_argument(
    "--test", type=str2bool, default=True, help="evaluation on test set."
)
parser.add_argument("--model", type=str, default="GCN_DGG", help="model name")
parser.add_argument(
    "--edge_noise_level",
    type=float,
    default=0.000,
    help="percentage of noisy edges to add",
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
    default="input_deg",
    choices=["pass", "input_deg", "gcn-x-deg", "x"],
    help="mode for the k_estimate_net in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--dgg_mode_k_select",
    type=str,
    default="k_only",
    choices=["edge_p-cdf", "k_only", "k_times_edge_prob"],
    help="mode for the k_selector in DGG, determines which features are used"
    "in the forward pass",
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


def train(args, model, optimizer, features, adj, labels, idx_train, device):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()

    # for name, p in model.dggs.named_parameters():
    #     print(name, p.grad.max().item(),  p.grad.mean().item(), p.grad.min().item())

    # if args.grad_clip != -1:
    #     torch.nn.utils.clip_grad_norm_(
    #         model.dggs.parameters(), max_norm=args.grad_clip
    #     )
    #
    # for name, p in model.dggs.named_parameters():
    #     print(name, p.grad.max().item(), p.grad.mean().item(),p.grad.min().item())

    optimizer.step()
    return loss_train.item(), acc_train.item()


def train_debug(
    args, model, optimizer, data, device, epoch, writer
):
    model.train()

    # parse data
    data = data.to(device)
    adj = to_scipy_sparse_matrix(
        edge_index=data.edge_index, num_nodes=data.num_nodes
    )
    if args.edge_noise_level > 0.0:
        adj = add_noisy_edges(adj, noise_level=args.edge_noise_level)

    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    features = data.x.to(device)
    labels = data.y.to(device)

    optimizer.zero_grad()
    output = model(features, adj, epoch, writer)
    acc_train = accuracy(output[data.train_mask], labels[data.train_mask].to(device))
    loss_train = F.nll_loss(output[data.train_mask], labels[data.train_mask].to(device))
    loss_train.backward()

    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate(model, data, device):
    model.eval()
    with torch.no_grad():
        adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        if args.edge_noise_level > 0.0:
            adj = add_noisy_edges(adj, noise_level=args.edge_noise_level)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        features = data.x.to(device)
        labels = data.y.to(device)

        output = model(features, adj)
        acc_val = accuracy(output[data.val_mask],
                             labels[data.val_mask].to(device))
        loss_val = F.nll_loss(output[data.val_mask],
                                labels[data.val_mask].to(device))

        return loss_val.item(), acc_val.item()


def test(model, data, device):
    model.eval()
    with torch.no_grad():
        adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index, num_nodes=data.num_nodes
        )
        if args.edge_noise_level > 0.0:
            adj = add_noisy_edges(adj, noise_level=args.edge_noise_level)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        features = data.x.to(device)
        labels = data.y.to(device)

        output = model(features, adj)
        acc_test = accuracy(output[data.test_mask],
                             labels[data.test_mask].to(device))
        loss_test = F.nll_loss(output[data.test_mask],
                                labels[data.test_mask].to(device))

        return loss_test.item(), acc_test.item()


def test_best(model, features, adj, labels, idx_test, checkpt_file, device):
    model.load_state_dict(torch.load(checkpt_file)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def load_data(args):
    if "DGG" not in args.model:
        args.pre_normalize_adj = False

    root = "/home/as03347/sceneEvolution/data/graph_data/{}".format(args.data)
    dataset = pygeo_datasets.__dict__['Planetoid'](
        root=root, name=args.data, split=args.split, transform=T.NormalizeFeatures()
    )
    data = dataset[0]

    return data, dataset


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
    data, dataset = load_data(args)
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
    best = 999999999
    best_epoch = 0
    acc = 0

    # writer.add_graph(model.cpu(), [features.cpu(), adj.to_dense().cpu()])

    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_debug(
            args,
            model,
            optimizer,
            data,
            device,
            epoch,
            writer,
        )
        loss_val, acc_val = validate(model, data, device)
        loss_test, acc_test = test(model, data, device)

        if (epoch + 1) % 1 == 0:
            print(
                "Epoch:{:04d}".format(epoch + 1),
                "train",
                "loss:{:.3f}".format(loss_tra),
                "acc:{:.2f}".format(acc_tra * 100),
                "| test",
                "loss:{:.3f}".format(loss_test),
                "acc:{:.2f}".format(acc_test * 100),
            )

            writer.add_scalar("train/loss", loss_tra, epoch)
            writer.add_scalar("train/acc", acc_tra, epoch)
            writer.add_scalar("val/loss", loss_val, epoch)
            writer.add_scalar("val/acc", acc_val, epoch)
            writer.add_scalar("test/acc", acc_test, epoch)

        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            # save_checkpoint(
            #     checkpt_file, args, epoch, model, optimizer, lr_scheduler="None"
            # )
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # if args.test:
    #     acc = test_best(model, features, adj, labels, idx_test, checkpt_file, device)[1]

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print("Load {}th epoch".format(best_epoch))
    print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100))
