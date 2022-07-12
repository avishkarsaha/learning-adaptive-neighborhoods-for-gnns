import os
import time
import random
import argparse
import glob
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import model as models
from utils import *
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = False


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/vol/research/sceneEvolution/models/GCNII",
        help="root directory",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default="cora_gcn_dgg_uvDist_edgePCDF",
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
    parser.add_argument("--layer", type=int, default=64, help="Number of layers.")
    parser.add_argument("--hidden", type=int, default=64, help="hidden dimensions.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.6,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument("--patience", type=int, default=2000, help="Patience")
    parser.add_argument("--data", default="cora", help="dateset")
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
        default=0.0,
        help="percentage of noisy edges to add",
    )
    # Differentiable graph generator specific
    parser.add_argument(
        "--dgm_dim",
        type=int,
        default=8,
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
    return parser.parse_args()


def load_karate_club():
    global A, x, labeled_nodes, labels

    def get_adj(noise_level=0.0):
        adj = torch.Tensor(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
            ]
        )

        # create noisy matrix
        if noise_level > 0.0:
            noise = torch.rand([adj.shape[0], adj.shape[1]])
            noise = (noise < noise_level).float()

            # mask out noise on current edges and diagonal
            mask = torch.ones(adj.shape)
            mask[torch.where(adj > 0)[0], torch.where(adj > 0)[1]] = 0
            mask[torch.arange(len(mask)), torch.arange(len(mask))] = 0
            noise = noise * mask

            adj = adj + noise
        if noise_level < 0.0:
            noise = torch.rand([adj.shape[0], adj.shape[1]])
            noise = (noise < -1 * noise_level).float()

            # only add noise on current edges
            mask = torch.zeros(adj.shape)
            mask[torch.where(adj > 0)[0], torch.where(adj > 0)[1]] = 1
            noise = noise * mask

            adj = adj - noise
            # new_adj[new_adj.sum(-1) == 0] = adj[new_adj.sum(-1) == 0]
            # adj = new_adj
        if noise_level == -1:
            adj = torch.zeros_like(adj)

        return adj.to_sparse()

    A = get_adj(noise_level=-0.9999).cuda()  # sparse unnormalized adjacency matrix
    x = torch.eye(A.shape[0]).cuda()
    labeled_nodes = torch.tensor(
        [0, 33]
    ).cuda()  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1]).cuda()
    return x, A, labels, labeled_nodes


def load_toy_dataset(k=5, mu_dist=2, n=100, noise_level=0.0, sparsity=0.0):
    torch.manual_seed(0)
    class_1 = torch.normal(
        mean=torch.zeros([n, 2]) + torch.tensor([5, 5]).unsqueeze(0),
        std=torch.ones([n, 2]) * 0.5,
    )
    class_2 = torch.normal(
        mean=torch.zeros([n, 2]) + torch.tensor([5 + mu_dist, 5]).unsqueeze(0),
        std=torch.ones([n, 2]) * 0.5,
    )

    # add specific cases
    ui_x = 5.0 + mu_dist / 2
    ui_y = 6.0
    ui = torch.tensor([ui_x, ui_y]).unsqueeze(0)  # [1, 2]
    r = 0.25
    theta = (torch.arange(10) / 10) * 2 * torch.pi
    vi_x = r * torch.cos(theta) + ui_x
    vi_y = r * torch.sin(theta) + ui_y
    vi = torch.stack([vi_x, vi_y]).T  # [10, 2]

    labels_ui = torch.tensor([0])
    labels_vi = (vi_x > ui_x).long()  # points on right of u are class 2 (label 0)

    # node features
    nodes = torch.cat([class_1, class_2, ui, vi], dim=0)
    mu = nodes.mean(0)
    std = nodes.std(0)
    nodes = (nodes - mu) / std
    labels = torch.cat(
        [torch.zeros(len(class_1)), torch.zeros(len(class_2)) + 1, labels_ui, labels_vi]
    ).long()

    # adjacency
    cdist = torch.cdist(nodes, nodes, p=2)
    topk_idxs = torch.topk(cdist, k=k, dim=-1, largest=False)[1]
    adj = torch.zeros_like(cdist)
    adj[torch.arange(len(adj)).unsqueeze(-1), topk_idxs] = 1
    adj[torch.arange(len(adj)), torch.arange(len(adj))] = 0

    # add noise
    if noise_level > 0.0:
        noise = torch.rand(adj.shape[0], adj.shape[1])
        noise_mask = torch.randint(low=0, high=2, size=adj.shape)
        diag_mask = 1 - torch.diag(torch.ones(len(adj)))
        noise_mask = noise_mask * diag_mask  # zero out diagonal

        # choose between current adjacency and noise with prob=noise_level
        choice = torch.rand(adj.shape[0], adj.shape[1])
        choice = (choice < noise_level).long()
        adj[choice > 0] = noise_mask[choice > 0]

        # sparsify by noise level
        sparsifier = torch.rand(adj.shape[0], adj.shape[1])
        sparsifier = (sparsifier > sparsity).float()
        adj = adj * sparsifier
        print("wait")

    # edges
    adj_sparse = adj.to_sparse()
    u = nodes[adj_sparse.indices()[0], :]
    v = nodes[adj_sparse.indices()[1], :]
    lines = [(a.numpy(), b.numpy()) for a, b in zip(u, v)]
    line_segs = LineCollection(lines, zorder=0, lw=0.5)

    fig, ax = plt.subplots()
    ax.add_collection(line_segs)
    ax.scatter(nodes[:, 0], nodes[:, 1], c=labels.numpy(), zorder=10)

    # for i in range(nodes.shape[0]):
    #     text_plot = ax.text(nodes[i, 0], nodes[i, 1], str(i))

    plt.title(
        "n = {}, k = {}, mu-dist = {}, noise = {}".format(n, k, mu_dist, noise_level)
    )
    plt.show()

    return nodes.cuda(), adj.to_sparse().cuda(), labels.cuda()


def run_experiment(model_name, k, mu_dist, n):
    print("exp {} {} {}".format(model_name, k, mu_dist))
    args = get_args()
    x, A, labels = load_toy_dataset(k=k, mu_dist=mu_dist, n=n)
    model = models.__dict__[model_name](nfeat=2, nhidden=5, nclass=2, args=args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 200
    accs = []
    for epoch in range(n_epochs):
        logits, _ = model(x, A, epoch)
        log_p = F.log_softmax(logits, dim=-1)

        # compute loss for labeled nodes
        # loss = F.nll_loss(log_p[labeled_nodes], labels)
        loss = F.nll_loss(log_p, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_label = torch.argmax(log_p, dim=-1)
        accuracy = (pred_label == labels).float().sum() / len(labels)

        wrong_pred_idx = torch.where(pred_label != labels)

        if epoch > n_epochs - 10:
            accs.append(accuracy)

        # if epoch % 19 == 0:
        #     print('Epoch %d | Loss: %.4f | Acc: %.3f' % (epoch, loss.item(), accuracy))
        # plt.scatter(logits.detach().cpu()[wrong_pred_idx][:, 0],
        #             logits.detach().cpu()[wrong_pred_idx][:, 1],
        #                 c=labels.cpu()[wrong_pred_idx], s=250)
        # plt.scatter(logits.detach().cpu()[:, 0], logits.detach().cpu()[:, 1],
        #                 c=labels.cpu(), alpha=0.9)
        # plt.title('Epoch %d Acc: %.3f' % (epoch, accuracy))
        # plt.show()

    final_acc = torch.mean(torch.tensor(accs))
    print("     final acc", final_acc)
    return final_acc


def run_tests(model_name, k, mu_dist):
    print("exp {} {} {}".format(model_name, k, mu_dist))
    args = get_args()
    x, A, labels = load_toy_dataset(k=k, mu_dist=mu_dist)
    model = models.__dict__[model_name](nfeat=2, nhidden=5, nclass=2, args=args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 200

    edge_q_ratio = edge_quality_discrete(A, labels)

    accs = []
    for epoch in range(n_epochs):
        logits, adj = model(x, A, epoch)
        log_p = F.log_softmax(logits, dim=-1)

        # compute loss for labeled nodes
        # loss = F.nll_loss(log_p[labeled_nodes], labels)
        loss = F.nll_loss(log_p, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_label = torch.argmax(log_p, dim=-1)
        accuracy = (pred_label == labels).float().sum() / len(labels)

        if epoch > n_epochs - 10:
            accs.append(accuracy)

        bad_ratio, good_ratio = edge_quality_cont(adj, labels)
        bad_ratio, good_ratio2 = edge_quality_cont2(adj, labels, A.to_dense())
        print(
            "ratio in v out {:.2f} {:.2f} {:.2f}".format(
                edge_q_ratio.mean().item(),
                good_ratio.mean().item(),
                adj.sum(-1).mean().item(),
            )
        )

        # if epoch % 19 == 0:
        #     print('Epoch %d | Loss: %.4f | Acc: %.3f' % (epoch, loss.item(), accuracy))
        # plt.scatter(logits.detach().cpu()[wrong_pred_idx][:, 0],
        #             logits.detach().cpu()[wrong_pred_idx][:, 1],
        #                 c=labels.cpu()[wrong_pred_idx], s=250)
        # plt.scatter(logits.detach().cpu()[:, 0], logits.detach().cpu()[:, 1],
        #                 c=labels.cpu(), alpha=0.9)
        # plt.title('Epoch %d Acc: %.3f' % (epoch, accuracy))
        # plt.show()

    final_acc = torch.mean(torch.tensor(accs))
    print("     final acc", final_acc)
    return final_acc


def run_vis_tests(model_name, k, mu_dist, n, noise, sparsity):
    print("exp {} {} {}".format(model_name, k, mu_dist))
    args = get_args()
    x, A, labels = load_toy_dataset(
        k=k, mu_dist=mu_dist, n=n, noise_level=noise, sparsity=sparsity
    )
    model = models.__dict__[model_name](nfeat=2, nhidden=5, nclass=2, args=args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 200

    edge_q_ratio = edge_quality_discrete(A, labels)

    accs = []
    for epoch in range(n_epochs):
        logits, var_dict = model(x, A, epoch)
        log_p = F.log_softmax(logits, dim=-1)

        # compute loss for labeled nodes
        loss = F.nll_loss(log_p, labels)
        optimizer.zero_grad()
        loss.backward()

        # accuracy
        pred_label = torch.argmax(log_p, dim=-1)
        accuracy = (pred_label == labels).float().sum() / len(labels)

        if epoch > n_epochs - 10:
            accs.append(accuracy)

        if epoch % 10 == 0:
            print("Epoch %d | Loss: %.4f | Acc: %.3f" % (epoch, loss.item(), accuracy))

        if epoch > n_epochs:
            fig = plt.figure(figsize=(5, 12))
            gs = fig.add_gridspec(7, 2)
            ax00 = fig.add_subplot(gs[0, 0])
            ax01 = fig.add_subplot(gs[1, 0])
            ax02 = fig.add_subplot(gs[2, 0])
            ax03 = fig.add_subplot(gs[3, 0])
            ax04 = fig.add_subplot(gs[4, 0])
            ax05 = fig.add_subplot(gs[5, 0])
            ax06 = fig.add_subplot(gs[6, 0])
            ax10 = fig.add_subplot(gs[0, 1])
            ax11 = fig.add_subplot(gs[1, 1])
            ax12 = fig.add_subplot(gs[2, 1])
            ax13 = fig.add_subplot(gs[3, 1])
            ax14 = fig.add_subplot(gs[4, 1])
            ax15 = fig.add_subplot(gs[5, 1])

            # variables to plot
            u_i = 40
            v_i = torch.where(A.to_dense()[u_i] == 1)

            n_neighbours = len(v_i[0])
            cmap = cm.get_cmap("hsv", n_neighbours)
            cgen = [cmap(i / n_neighbours) for i in range(n_neighbours)]

            u = x[u_i].cpu().detach()
            v = x[v_i].cpu().detach()
            lines = [(u.cpu().numpy(), b.cpu().numpy()) for b in v]
            line_segs = LineCollection(lines, zorder=0, color=cgen)

            u_v_dist = torch.linalg.norm(u.unsqueeze(0) - v, dim=-1, ord=2)
            edge_prob = var_dict["edge_p"].squeeze(0)[u_i].cpu().detach()
            first_k = var_dict["first_k"].squeeze(0)[u_i].cpu().detach()
            out_adj = var_dict["out_adj"].squeeze(0)[u_i].cpu().detach()
            s_edge_p, s_edge_p_idxs = torch.sort(edge_prob, descending=True)
            line_segs_out = LineCollection(
                lines, zorder=0, lw=out_adj[v_i] * 10, color=cgen
            )
            u_logits = logits[u_i].cpu().detach().numpy()
            v_logits = logits[v_i].cpu().detach().numpy()
            logits_lines = [(u_logits, b) for b in v_logits]
            logits_segs = LineCollection(
                logits_lines, zorder=0, lw=out_adj[v_i] * 10, color=cgen
            )
            u_label = pred_label[u_i].detach().cpu()
            v_label = pred_label[v_i].detach().cpu()

            # variable gradients to plot
            grad_edge_prob = (
                model.dggs[0].var_grads["edge_p"][0].squeeze(0)[u_i].cpu().detach()
            )
            grad_actual_k = (
                model.dggs[0].var_grads["actual_k"][0].squeeze(0)[u_i].cpu().detach()
            )
            grad_out_adj = (
                model.dggs[0].var_grads["out_adj"][0].squeeze(0)[u_i].cpu().detach()
            )
            grad_s_edge_p = grad_edge_prob[s_edge_p_idxs]

            ax00.add_collection(line_segs)
            ax00.scatter(u[0], u[1], c=labels[u_i].cpu().detach())
            ax00.scatter(v[:, 0], v[:, 1], c=labels[v_i].cpu().detach())

            ax01.bar(x=np.arange(len(u_v_dist)), height=u_v_dist, color=cgen)
            ax02.bar(
                x=np.arange(len(edge_prob[v_i])), height=edge_prob[v_i], color=cgen
            )
            ax03.plot(np.arange(len(first_k)), first_k)

            ax04.plot(np.arange(len(first_k)), first_k)
            ax04.bar(x=np.arange(len(edge_prob)), height=s_edge_p)

            ax05.add_collection(logits_segs)
            ax05.scatter(u_logits[0], u_logits[1], c=u_label)
            ax05.scatter(v_logits[:, 0], v_logits[:, 1], c=v_label)

            ax06.add_collection(line_segs_out)
            ax06.scatter(u[0], u[1], c=u_label)
            ax06.scatter(v[:, 0], v[:, 1], c=v_label)

            ax12.bar(x=np.arange(len(edge_prob)), height=grad_edge_prob, color=cgen)
            ax13.bar(x=1, height=grad_actual_k)
            ax14.bar(x=np.arange(len(edge_prob)), height=grad_s_edge_p, color=cgen)

            ax00.set_title("Input Graph, node i")
            ax01.set_title("Distances of neighbours")
            ax02.set_title("Edge probabilities")
            ax03.set_title("Sigmoid with inflection at K")
            ax04.set_title("Sorted Edge prob * K curve")
            ax05.set_title("Pred. labels + edges in emb. space")
            ax06.set_title("Pred. labels + edges in input space")

            # plt.show()
            # fig.tight_layout()
            # os.makedirs('/vol/research/sceneEvolution/models/GCNII/visualisations'
            #         '/n{}_k{}_mudist{}_noise{}'.format(n, k, mu_dist, noise), exist_ok=True)
            #
            # plt.savefig(
            #     os.path.join(
            #         '/vol/research/sceneEvolution/models/GCNII/visualisations'
            #         '/n{}_k{}_mudist{}_noise{}/{:04d}.png'.format(n, k, mu_dist, noise, epoch)
            #     )
            # )
            # fig.clf()
            # plt.close(fig)

        optimizer.step()

    final_acc = torch.mean(torch.tensor(accs))
    print("     final acc", final_acc)
    return final_acc


def edge_quality_cont(adj, labels):
    deg = adj.sum(-1)
    l_id = labels + 8
    good_ratio = (l_id.unsqueeze(1) == l_id.unsqueeze(0)).float()
    good_ratio = good_ratio * adj
    good_e = good_ratio.sum(-1)
    print("{:.3f} {:.3f}".format(deg.mean().item(), good_e.mean().item()))

    bad_ratio = l_id.unsqueeze(1) != l_id.unsqueeze(0)
    bad_ratio = bad_ratio * adj
    bad_e = bad_ratio.sum(-1)

    good_ratio = good_e / deg
    bad_ratio = bad_e / deg
    return bad_ratio, good_ratio


def edge_quality_cont2(adj, labels, mask):
    deg = mask.sum(-1)
    l_id = labels + 8
    good_ratio = (l_id.unsqueeze(1) == l_id.unsqueeze(0)).float()
    good_ratio = good_ratio * mask  # only look at ratio relative to original input adj
    good_ratio = good_ratio * adj
    good_e = good_ratio.sum(-1)

    bad_ratio = l_id.unsqueeze(1) != l_id.unsqueeze(0)
    bad_ratio = bad_ratio * adj
    bad_e = bad_ratio.sum(-1)

    good_ratio = good_e / deg
    bad_ratio = bad_e / deg
    return bad_ratio, good_ratio


def edge_quality_discrete(A, labels):
    """
    calculate ratio of good edges for each node (ie edges to neighbours which match
    the ego node's ground truth class)
    Args:
        A:
        labels:

    Returns:

    """
    deg = A.to_dense().sum(-1)
    l_id = labels + 8
    ratio = A.to_dense() * l_id.unsqueeze(0)
    good_e = (ratio == l_id.unsqueeze(1)).sum(-1)
    good_ratio = good_e / deg
    return good_ratio


def make_gif():
    print("making gif")
    # Get images
    image_dir = (
        "/vol/research/sceneEvolution/models/GCNII/visualisations/n20_k10_mudist1"
    )
    save_dir = "/vol/research/sceneEvolution/models/GCNII/visualisations/"
    image_path = os.path.join(image_dir, "*.png")
    gif_path = os.path.join(save_dir, "n20_k10_mudist1.gif")
    image_path = sorted(glob.glob(image_path))

    img, *imgs = [Image.open(f) for f in image_path]
    img.save(
        fp=gif_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=1500,
        loop=0,
    )
    print("done")


if __name__ == "__main__":
    # K = [2, 5, 10, 20, 50]
    # MU = [0., 0.25, 0.5, 1.0, 2.0]
    # accs = torch.zeros([len(K), len(MU)])
    # model = 'GCN_DGG_debug'
    #
    # for i, k in enumerate(K):
    #     for j, mu in enumerate(MU):
    #         acc = run_experiment(model_name=model, k=k, mu_dist=mu, n=50)
    #         accs[i, j] = acc
    #
    # plt.imshow(accs.T, vmax=1.0, vmin=0.5, cmap='plasma')
    # plt.xticks(ticks=np.arange(len(K)), labels=K)
    # plt.yticks(ticks=np.arange(len(MU)), labels=MU)
    # plt.title(model)
    # plt.show()

    # run_tests(model_name='GCN_DGG_debug', k=100, mu_dist=2)
    run_vis_tests(
        model_name="GCN_debug", k=5, mu_dist=1, n=100, noise=1.0, sparsity=0.9
    )
    # make_gif()
