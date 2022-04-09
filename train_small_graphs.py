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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/vol/research/sceneEvolution/models/GCNII",
    help="root directory",
)
parser.add_argument(
    "--expname", type=str, default="debug_small_graphs_dgg", help="experiment name"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=100, help="Number of epochs to train."
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
parser.add_argument("--patience", type=int, default=100, help="Patience")
parser.add_argument("--data", default="cora", help="dateset")
parser.add_argument("--dev", type=int, default=0, help="device id")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha_l")
parser.add_argument("--lamda", type=float, default=0.5, help="lamda.")
parser.add_argument("--variant", type=str2bool, default=False, help="GCN* model.")
parser.add_argument(
    "--test", type=str2bool, default=True, help="evaluation on test set."
)
parser.add_argument(
    "--model", type=str, default='GCNII_DGG', help="model name"
)
# Differentiable graph generator specific
parser.add_argument(
    "--graph_gen_space",
    type=str,
    default="embedding",
    choices=["input", "embedding"],
    help="euclidean space of nodes in which to calculate adjacency matrix",
)
parser.add_argument(
    "--dgm_dim",
    type=int,
    default=128,
    help="Dimensions of the linear projection layer in the DGM",
)
parser.add_argument(
    "--spatial-k",
    type=int,
    default=2,
    help="number of nearest neighbours in space in DGM",
)
parser.add_argument(
    "--self_loops_noise",
    type=str2bool,
    default=False,
    help="Whether to add gumbel noise to self loop edge probabilities",
)
parser.add_argument(
    "--st_gumbel_softmax",
    type=str2bool,
    default=True,
    help="Whether to do straight through gumbel softmax "
    "(argmax in forward, softmax in backward) or just softmax top k in both",
)
parser.add_argument(
    "--dgm_temp",
    type=float,
    default=2,
    help="Gumvel softmax temperature",
)
parser.add_argument(
    "--test_noise",
    type=str2bool,
    default=False,
    help="Whether to add noise to when sampling at test time",
)
parser.add_argument(
    "--dgg_dist_fn",
    type=str,
    default="metric",
    help="distance function for calculating edge probabilities",
)
parser.add_argument(
    "--k_bias",
    type=float,
    default=1.0,
    help="bias to add to predicted K value",
)
parser.add_argument(
    "--k_net_input",
    type=str,
    default="raw",
    help="input features to Learnable K network",
)
parser.add_argument(
    "--hs_start",
    type=float,
    default=1.25,
    help="heaviside project points x start",
)
parser.add_argument(
    "--hs_end",
    type=float,
    default=-1.25,
    help="heaviside project points x end",
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


def train(model, optimizer, features, adj, labels, idx_train, device):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate(model, features, adj, labels, idx_val, device):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test(model, features, adj, labels, idx_test, device):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_val = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_val.item(), acc_val.item()


def test_best(model, features, adj, labels, idx_test, checkpt_file, device):
    model.load_state_dict(torch.load(checkpt_file)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


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
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    checkpt_file = os.path.join(expdir, uuid.uuid4().hex + ".pt")
    print(checkpt_file)

    # Tensorboard writer
    writer = SummaryWriter(tbdir)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_citation(
        args.data, args.root
    )
    cudaid = "cuda"
    device = torch.device(cudaid)
    features = features.to(device)
    adj = adj.to(device)

    # Load model
    model = models.__dict__[args.model](
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=args.variant,
        args=args
    ).to(device)

    optimizer = optim.Adam(
        [
            {"params": model.params1, "weight_decay": args.wd1},
            {"params": model.params2, "weight_decay": args.wd2},
        ],
        lr=args.lr,
    )

    # Run
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train(
            model, optimizer, features, adj, labels, idx_train, device
        )
        loss_val, acc_val = validate(model, features, adj, labels, idx_val, device)
        acc_test = test(model, features, adj, labels, idx_test, device)[1]

        if (epoch + 1) % 1 == 0:
            print(
                "Epoch:{:04d}".format(epoch + 1),
                "train",
                "loss:{:.3f}".format(loss_tra),
                "acc:{:.2f}".format(acc_tra * 100),
                "| val",
                "loss:{:.3f}".format(loss_val),
                "acc:{:.2f}".format(acc_val * 100),
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
            save_checkpoint(
                checkpt_file, args, epoch, model, optimizer, lr_scheduler="None"
            )
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        acc = test_best(model, features, adj, labels, idx_test, checkpt_file, device)[1]

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print("Load {}th epoch".format(best_epoch))
    print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100))
