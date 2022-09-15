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
from sklearn.metrics import f1_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/vol/research/sceneEvolution/models/GCNII",
    help="root directory",
)
parser.add_argument(
    "--expname", type=str, default="220827_ppi_gcn", help="experiment name"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=8000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument(
    "--wd1", type=float, default=0.01, help="weight decay (L2 loss on parameters)."
)
parser.add_argument(
    "--wd2", type=float, default=5e-4, help="weight decay (L2 loss on parameters)."
)
parser.add_argument("--layer", type=int, default=9, help="Number of hidden layers.")
parser.add_argument("--hidden", type=int, default=2048, help="Number of hidden layers.")
parser.add_argument(
    "--dropout", type=float, default=0.2, help="Dropout rate (1 - keep probability)."
)
parser.add_argument("--patience", type=int, default=2000, help="Patience")
parser.add_argument("--data", default="ppi", help="dateset")
parser.add_argument("--dev", type=int, default=0, help="device id")
parser.add_argument("--alpha", type=float, default=0.5, help="alpha_l")
parser.add_argument("--lamda", type=float, default=1, help="lamda.")
parser.add_argument("--variant", type=str2bool, default=False, help="GCN* model.")
parser.add_argument(
    "--test", type=str2bool, default=True, help="evaluation on test set."
)
parser.add_argument("--model", type=str, default="GCN_MultiClass", help="model name")
parser.add_argument(
    "--edge_noise_level",
    type=float,
    default=0.000,
    help="percentage of noisy edges to add",
)
parser.add_argument(
    "--remove_interclass_edges",
    type=float,
    default=1.0,
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
    default=1,
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
    default="project_adj",
    help="mode for the edge_prob_net in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--dgg_mode_k_net",
    type=str,
    default="learn_normalized_degree_relu",
    help="mode for the k_estimate_net in DGG, determines which features are used"
    "in the forward pass",
)
parser.add_argument(
    "--dgg_mode_k_select",
    type=str,
    default="k_only",
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


def train(
    args,
    model,
    optimizer,
    train_feat,
    train_adj,
    train_labels,
    train_nodes,
    loss_fcn,
    device,
    writer,
    epoch,
):
    model.train()
    loss_tra = 0
    acc_tra = 0
    for step, batch in enumerate(loader):
        batch_adj = train_adj[batch[0]].to(device)
        batch_feature = train_feat[batch[0]].to(device)
        batch_label = train_labels[batch[0]].to(device)

        # get adjacency with interclass edges removed
        if args.remove_interclass_edges > 0:
            batch_adj = remove_multi_interclass_edges(batch_adj, batch_label)

        optimizer.zero_grad()
        output, out_adj, x_dgg = model(batch_feature, batch_adj, writer=writer, epoch=epoch)
        loss_train = loss_fcn(
            output[: train_nodes[batch]], batch_label[: train_nodes[batch]]
        )
        loss_train.backward()
        optimizer.step()
        loss_tra += loss_train.item()
    loss_tra /= 20
    acc_tra /= 20
    return loss_tra, acc_tra


def validate(args, model, val_feat, val_adj, val_labels, val_nodes, loss_fcn, device):
    loss_val = 0
    acc_val = 0
    for batch in range(2):
        batch_adj = val_adj[batch].to(device)
        batch_feature = val_feat[batch].to(device)
        batch_label = val_labels[batch].to(device)
        score, val_loss = evaluate(
            batch_feature, model, val_nodes[batch], batch_adj, batch_label, loss_fcn
        )
        loss_val += val_loss
        acc_val += score
    loss_val /= 2
    acc_val /= 2
    return loss_val, acc_val


def test(args, model, test_feat, test_adj, test_labels, test_nodes, loss_fcn, device):
    model.eval()
    loss_test = 0
    acc_test = 0
    for batch in range(2):
        batch_adj = test_adj[batch].to(device)
        batch_feature = test_feat[batch].to(device)
        batch_label = test_labels[batch].to(device)

        # get adjacency with interclass edges removed
        if args.remove_interclass_edges > 0:
            batch_adj = remove_multi_interclass_edges(batch_adj, batch_label)

        score, loss = evaluate(
            batch_feature, model, test_nodes[batch], batch_adj, batch_label, loss_fcn
        )
        loss_test += loss
        acc_test += score
    acc_test /= 2
    loss_test /= 2
    return loss_test, acc_test


# adapted from DGL
def evaluate(feats, model, idx, subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        output, out_adj, x_dgg = model(feats, subgraph)
        loss_data = loss_fcn(output[:idx], labels[:idx].float())
        predict = np.where(output[:idx].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels[:idx].data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


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
    (
        train_adj,
        val_adj,
        test_adj,
        train_feat,
        val_feat,
        test_feat,
        train_labels,
        val_labels,
        test_labels,
        train_nodes,
        val_nodes,
        test_nodes,
    ) = load_ppi_from_disk(normalize_adj=True)

    idx = torch.LongTensor(range(20))
    loader = Data.DataLoader(dataset=idx, batch_size=1, shuffle=True, num_workers=0)

    cudaid = "cuda"
    device = torch.device(cudaid)

    # Load model
    model = models.__dict__[args.model](
        nfeat=train_feat[0].shape[1],
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=train_labels[0].shape[1],
        dropout=args.dropout,
        lamda=args.lamda,
        alpha=args.alpha,
        variant=args.variant,
        args=args,
    ).to(device)
    loss_fcn = torch.nn.BCELoss()

    if "GCN" in args.model and "II" in args.model:
        optimizer = optim.Adam(
            [
                {"params": model.params1, "weight_decay": args.wd1},
                {"params": model.params2, "weight_decay": args.wd2},
            ],
            lr=args.lr,
        )
    elif "GCN" in args.model and "II" not in args.model:
        optimizer = optim.Adam(
            [
                dict(params=model.params1, weight_decay=5e-4),
                dict(params=model.params2, weight_decay=0),
            ],
            lr=args.lr,
        )  # Only perform weight-decay on first convolution.
    elif "SAGE" in args.model:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif 'GAT' in args.model:
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif 'GCN_MultiClass' in args.model:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Run
    t_total = time.time()
    bad_counter = 0
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train(
            args,
            model,
            optimizer,
            train_feat,
            train_adj,
            train_labels,
            train_nodes,
            loss_fcn,
            device,
            writer,
            epoch,
        )
        loss_val, acc_val = validate(
            args, model, val_feat, val_adj, val_labels, val_nodes, loss_fcn, device
        )
        acc_test = test(
            args, model, test_feat, test_adj, test_labels, test_nodes, loss_fcn, device
        )[1]

        if (epoch + 1) % 1 == 0:
            print(
                "Epoch:{:04d}".format(epoch + 1),
                "train",
                "loss:{:.3f}".format(loss_tra),
                "| val",
                "loss:{:.3f}".format(loss_val),
                "f1:{:.3f}".format(acc_val * 100),
            )

            writer.add_scalar("train/loss", loss_tra, epoch)
            writer.add_scalar("train/acc", acc_tra, epoch)
            writer.add_scalar("val/loss", loss_val, epoch)
            writer.add_scalar("val/acc", acc_val, epoch)
            writer.add_scalar("test/acc", acc_test, epoch)

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print("Load {}th epoch".format(best_epoch))
    print("Test" if args.test else "Val", "f1.:{:.2f}".format(acc * 100))
