import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GCN_debug, GCN_DGG


def load_karate_club():
    global A, x, labeled_nodes, labels

    def get_adj(noise_level=0.0):
        adj = torch.Tensor(
            [[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
             [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
              1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
              1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
             ])

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
        [0, 33]).cuda()  # only the instructor and the president nodes are labeled
    labels = torch.tensor([0, 1]).cuda()
    return x, A, labels, labeled_nodes

def load_toy_dataset():
    torch.manual_seed(0)
    n = 100
    class_1 = torch.normal(
        mean=torch.zeros([n, 2]) + torch.tensor([5, 5]).unsqueeze(0),
        std=torch.ones([n, 2]) * 0.5
    )
    class_2 = torch.normal(
        mean=torch.zeros([n, 2]) + torch.tensor([7, 5]).unsqueeze(0),
        std=torch.ones([n, 2]) * 0.5
    )

    plt.scatter(class_1[:, 0], class_1[:, 1], c='b')
    plt.scatter(class_2[:, 0], class_2[:, 1], c='r')
    plt.show()

load_toy_dataset()
exit()
x, A, labels, labeled_nodes = load_karate_club()

model = GCN_debug(nfeat=34, nhidden=5, nclass=2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
all_logits = []

for epoch in range(80):
    logits = model(x, A)
    log_p = F.log_softmax(logits, dim=-1)

    # compute loss for labeled nodes
    loss = F.nll_loss(log_p[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred_label = torch.argmax(log_p, dim=-1)
    accuracy = (pred_label[labeled_nodes] == labels).float().sum() / len(labels)
    print('Epoch %d | Loss: %.4f | Acc: %.3f' % (epoch, loss.item(), accuracy))

    if epoch % 20 == 0:
        plt.scatter(logits.detach().cpu()[:, 0], logits.detach().cpu()[:, 1],
                    c=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0
                        , 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        plt.show()
