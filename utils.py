import os.path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
from torch_geometric.datasets import AttributedGraphDataset
sys.setrecursionlimit(99999)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def torch_normalized_adjacency(adj, mode='add_self_loops'):
    if mode == 'add_self_loops':
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        row_sum = adj.sum(1)
        row_sum = (row_sum == 0) * 1 + row_sum
        d_inv_sqrt = (row_sum ** -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj
    elif mode == 'self_loops_present':
        # adj = adj + torch.eye(adj.shape[0], device=adj.device)
        row_sum = adj.sum(1)
        row_sum = (row_sum == 0) * 1 + row_sum
        d_inv_sqrt = (row_sum ** -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_noisy_edges(adj, noise_level=0.1):
    np.random.seed(0)
    adj = sp.coo_matrix(adj)

    # create noisy matrix
    noise = np.random.rand(adj.shape[0], adj.shape[1])
    noise = (noise < noise_level).astype(np.float)

    # mask out noise on current edges and diagonal
    mask = np.ones(adj.shape)
    mask[adj.row, adj.col] = 0
    mask[np.arange(len(mask)), np.arange(len(mask))] = 0
    noise = noise * mask

    noisy_adj = adj + noise
    noisy_adj = sp.csr_matrix(noisy_adj)
    return noisy_adj


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# adapted from tkipf/gcn
def load_citation(dataset_str, root, normalize_adj=False, noise=0.0):
    """
    Load Citation Networks Datasets.
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    datadir = os.path.join(root, "data")
    for i in range(len(names)):
        with open(
            os.path.join(datadir, "ind.{}.{}".format(dataset_str.lower(), names[i])),
            "rb",
        ) as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    features = normalize(features)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # zero out self loops if there are any (these will be added in the network)
    adj.setdiag(np.zeros(adj.shape[0]), k=0)
    assert adj.diagonal().sum() == 0

    # add noise (optional)
    if noise > 0.0:
        adj = add_noisy_edges(adj, noise_level=noise)

    # normalize adjacency if not using DGG

    if normalize_adj:
        adj = sys_normalized_adjacency(adj) # adds self loops and normalises

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # norm_adj = sys_normalized_adjacency(adj)    # adds self loops and normalises
    # norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
    #
    #
    # norm_adj_gcn = normalize_adj_gcn(adj.to_dense())
    # norm_norm_adj_gcn = normalize_adj_gcn(norm_adj.to_dense())

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj_gcn(A):
    # add self loops
    A_hat = A + torch.eye(A.size(0))
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    return A_hat



# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        # for v in range(nb_nodes):
        for v in adj[u, :].nonzero()[1]:
            # if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)


def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret


def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
                #  if adj[i,j] == 1:
                return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits = {}
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] == 0 or mapping[j] == 0:
                dict_splits[0] = None
            elif mapping[i] == mapping[j]:
                if (
                    ds_label[i]["val"] == ds_label[j]["val"]
                    and ds_label[i]["test"] == ds_label[j]["test"]
                ):

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]["val"]:
                            dict_splits[mapping[i]] = "val"

                        elif ds_label[i]["test"]:
                            dict_splits[mapping[i]] = "test"

                        else:
                            dict_splits[mapping[i]] = "train"

                    else:
                        if ds_label[i]["test"]:
                            ind_label = "test"
                        elif ds_label[i]["val"]:
                            ind_label = "val"
                        else:
                            ind_label = "train"
                        if dict_splits[mapping[i]] != ind_label:
                            print("inconsistent labels within a graph exiting!!!")
                            return None
                else:
                    print("label of both nodes different, exiting!!")
                    return None
    return dict_splits


def load_ppi(normalize_adj=True):

    print("Loading G...")
    with open("ppi/ppi-G.json") as jsonfile:
        g_data = json.load(jsonfile)
    # print (len(g_data))
    G = json_graph.node_link_graph(g_data)

    # Extracting adjacency matrix
    adj = nx.adjacency_matrix(G)

    prev_key = ""
    for key, value in g_data.items():
        if prev_key != key:
            # print (key)
            prev_key = key

    # print ('Loading id_map...')
    with open("ppi/ppi-id_map.json") as jsonfile:
        id_map = json.load(jsonfile)
    # print (len(id_map))

    id_map = {int(k): int(v) for k, v in id_map.items()}
    for key, value in id_map.items():
        id_map[key] = [value]
    # print (len(id_map))

    print("Loading features...")
    features_ = np.load("ppi/ppi-feats.npy")
    # print (features_.shape)

    # standarizing features
    from sklearn.preprocessing import StandardScaler

    train_ids = np.array(
        [
            id_map[n]
            for n in G.nodes()
            if not G.nodes[n]["val"] and not G.nodes[n]["test"]
        ]
    )
    train_feats = features_[train_ids[:, 0]]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)

    features = sp.csr_matrix(features_).tolil()

    print("Loading class_map...")
    class_map = {}
    with open("ppi/ppi-class_map.json") as jsonfile:
        class_map = json.load(jsonfile)
    # print (len(class_map))

    # pdb.set_trace()
    # Split graph into sub-graphs
    # print ('Splitting graph...')
    splits = dfs_split(adj)

    # Rearrange sub-graph index and append sub-graphs with 1 or 2 nodes to bigger sub-graphs
    # print ('Re-arranging sub-graph IDs...')
    list_splits = splits.tolist()
    group_inc = 1

    for i in range(np.max(list_splits) + 1):
        if list_splits.count(i) >= 3:
            splits[np.array(list_splits) == i] = group_inc
            group_inc += 1
        else:
            # splits[np.array(list_splits) == i] = 0
            ind_nodes = np.argwhere(np.array(list_splits) == i)
            ind_nodes = ind_nodes[:, 0].tolist()
            split = None

            for ind_node in ind_nodes:
                if g_data["nodes"][ind_node]["val"]:
                    if split is None or split == "val":
                        splits[np.array(list_splits) == i] = 21
                        split = "val"
                    else:
                        raise ValueError(
                            "new node is VAL but previously was {}".format(split)
                        )
                elif g_data["nodes"][ind_node]["test"]:
                    if split is None or split == "test":
                        splits[np.array(list_splits) == i] = 23
                        split = "test"
                    else:
                        raise ValueError(
                            "new node is TEST but previously was {}".format(split)
                        )
                else:
                    if split is None or split == "train":
                        splits[np.array(list_splits) == i] = 1
                        split = "train"
                    else:
                        pdb.set_trace()
                        raise ValueError(
                            "new node is TRAIN but previously was {}".format(split)
                        )

    # counting number of nodes per sub-graph
    list_splits = splits.tolist()
    nodes_per_graph = []
    for i in range(1, np.max(list_splits) + 1):
        nodes_per_graph.append(list_splits.count(i))

    # Splitting adj matrix into sub-graphs
    subgraph_nodes = np.max(nodes_per_graph)
    adj_sub = np.empty((len(nodes_per_graph), subgraph_nodes, subgraph_nodes))
    feat_sub = np.empty((len(nodes_per_graph), subgraph_nodes, features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), subgraph_nodes, 121))

    for i in range(1, np.max(list_splits) + 1):
        # Creating same size sub-graphs
        indexes = np.where(splits == i)[0]
        subgraph_ = adj[indexes, :][:, indexes]

        if subgraph_.shape[0] < subgraph_nodes or subgraph_.shape[1] < subgraph_nodes:
            subgraph = np.identity(subgraph_nodes)
            feats = np.zeros([subgraph_nodes, features.shape[1]])
            labels = np.zeros([subgraph_nodes, 121])
            # adj
            subgraph = sp.csr_matrix(subgraph).tolil()
            subgraph[0 : subgraph_.shape[0], 0 : subgraph_.shape[1]] = subgraph_
            adj_sub[i - 1, :, :] = subgraph.todense()
            # feats
            feats[0 : len(indexes)] = features[indexes, :].todense()
            feat_sub[i - 1, :, :] = feats
            # labels
            for j, node in enumerate(indexes):
                labels[j, :] = np.array(class_map[str(node)])
            labels[indexes.shape[0] : subgraph_nodes, :] = np.zeros([121])
            labels_sub[i - 1, :, :] = labels

        else:
            adj_sub[i - 1, :, :] = subgraph_.todense()
            feat_sub[i - 1, :, :] = features[indexes, :].todense()
            for j, node in enumerate(indexes):
                labels[j, :] = np.array(class_map[str(node)])
            labels_sub[i - 1, :, :] = labels

    # Get relation between id sub-graph and tran,val or test set
    dict_splits = find_split(adj, splits, g_data["nodes"])

    # Testing if sub graphs are isolated
    # print ('Are sub-graphs isolated?')
    # print (test(adj, splits))

    # Splitting tensors into train,val and test
    train_split = []
    val_split = []
    test_split = []
    for key, value in dict_splits.items():
        if dict_splits[key] == "train":
            train_split.append(int(key) - 1)
        elif dict_splits[key] == "val":
            val_split.append(int(key) - 1)
        elif dict_splits[key] == "test":
            test_split.append(int(key) - 1)

    train_adj = adj_sub[train_split, :, :]
    val_adj = adj_sub[val_split, :, :]
    test_adj = adj_sub[test_split, :, :]

    train_feat = feat_sub[train_split, :, :]
    val_feat = feat_sub[val_split, :, :]
    test_feat = feat_sub[test_split, :, :]

    train_labels = labels_sub[train_split, :, :]
    val_labels = labels_sub[val_split, :, :]
    test_labels = labels_sub[test_split, :, :]

    train_nodes = np.array(nodes_per_graph[train_split[0] : train_split[-1] + 1])
    val_nodes = np.array(nodes_per_graph[val_split[0] : val_split[-1] + 1])
    test_nodes = np.array(nodes_per_graph[test_split[0] : test_split[-1] + 1])

    # Masks with ones

    tr_msk = np.zeros(
        (len(nodes_per_graph[train_split[0] : train_split[-1] + 1]), subgraph_nodes)
    )
    vl_msk = np.zeros(
        (len(nodes_per_graph[val_split[0] : val_split[-1] + 1]), subgraph_nodes)
    )
    ts_msk = np.zeros(
        (len(nodes_per_graph[test_split[0] : test_split[-1] + 1]), subgraph_nodes)
    )

    for i in range(len(train_nodes)):
        for j in range(train_nodes[i]):
            tr_msk[i][j] = 1

    for i in range(len(val_nodes)):
        for j in range(val_nodes[i]):
            vl_msk[i][j] = 1

    for i in range(len(test_nodes)):
        for j in range(test_nodes[i]):
            ts_msk[i][j] = 1

    train_adj_list = []
    val_adj_list = []
    test_adj_list = []
    for i in range(train_adj.shape[0]):
        adj = sp.coo_matrix(train_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if normalize_adj:
            adj = sys_normalized_adjacency(adj)
        # tmp = sys_normalized_adjacency(adj)
        train_adj_list.append(sparse_mx_to_torch_sparse_tensor(adj))
    for i in range(val_adj.shape[0]):
        adj = sp.coo_matrix(val_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if normalize_adj:
            adj = sys_normalized_adjacency(adj)
        val_adj_list.append(sparse_mx_to_torch_sparse_tensor(adj))
        adj = sp.coo_matrix(test_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if normalize_adj:
            adj = sys_normalized_adjacency(adj)
        test_adj_list.append(sparse_mx_to_torch_sparse_tensor(adj))

    train_feat = torch.FloatTensor(train_feat)
    val_feat = torch.FloatTensor(val_feat)
    test_feat = torch.FloatTensor(test_feat)

    train_labels = torch.FloatTensor(train_labels)
    val_labels = torch.FloatTensor(val_labels)
    test_labels = torch.FloatTensor(test_labels)

    tr_msk = torch.LongTensor(tr_msk)
    vl_msk = torch.LongTensor(vl_msk)
    ts_msk = torch.LongTensor(ts_msk)

    # save_fn = '/vol/research/sceneEvolution/models/GCNII/' \
    #           'ppi/ppi_data_adj_norm{}.pt'.format(str(normalize_adj))
    # torch.save(
    #     {
    #         'train_adj_list': train_adj_list,
    #         'val_adj_list': val_adj_list,
    #         'test_adj_list': test_adj_list,
    #         'train_feat': train_feat,
    #         'val_feat': val_feat,
    #         'test_feat': test_feat,
    #         'train_labels': train_labels,
    #         'val_labels': val_labels,
    #         'test_labels': test_labels,
    #         'train_nodes': train_nodes,
    #         'val_nodes': val_nodes,
    #         'test_nodes': test_nodes,
    #     },
    #     save_fn
    # )
    # exit()

    return (
        train_adj_list,
        val_adj_list,
        test_adj_list,
        train_feat,
        val_feat,
        test_feat,
        train_labels,
        val_labels,
        test_labels,
        train_nodes,
        val_nodes,
        test_nodes,
    )

def load_ppi_from_disk(normalize_adj=True):

    fn = '/vol/research/sceneEvolution/models/GCNII/' \
              'ppi/ppi_data_adj_norm{}.pt'.format(str(normalize_adj))
    data = torch.load(fn)

    train_adj_list = data['train_adj_list']
    val_adj_list = data['val_adj_list']
    test_adj_list = data['test_adj_list']
    train_feat = data['train_feat']
    val_feat = data['val_feat']
    test_feat = data['test_feat']
    train_labels = data['train_labels']
    val_labels = data['val_labels']
    test_labels = data['test_labels']
    train_nodes = data['train_nodes']
    val_nodes = data['val_nodes']
    test_nodes = data['test_nodes']

    return (
        train_adj_list,
        val_adj_list,
        test_adj_list,
        train_feat,
        val_feat,
        test_feat,
        train_labels,
        val_labels,
        test_labels,
        train_nodes,
        val_nodes,
        test_nodes,
    )

def load_data(path, name):
    # dataset = AttributedGraphDataset(root=root, name=name)
    # print('wait')

    import copy
    import os.path as osp

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from torch_geometric.datasets import Reddit
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Reddit(path)

    # Already send node features/labels to GPU for faster access during sampling:
    data = dataset[0].to(device, 'x', 'y')

    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                  num_neighbors=[25, 10], shuffle=True, **kwargs)

    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                     num_neighbors=[-1], shuffle=False, **kwargs)

    # No need to maintain these features during evaluation:
    del subgraph_loader.data.x, subgraph_loader.data.y
    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    class SAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                    x = F.dropout(x, p=0.5, training=self.training)
            return x

        @torch.no_grad()
        def inference(self, x_all, subgraph_loader):
            pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
            pbar.set_description('Evaluating')

            # Compute representations of nodes layer by layer, using *all*
            # available edges. This leads to faster computation in contrast to
            # immediately computing the final representations of each batch:
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:
                    x = x_all[batch.n_id.to(x_all.device)].to(device)
                    x = conv(x, batch.edge_index.to(device))
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                    pbar.update(batch.batch_size)
                x_all = torch.cat(xs, dim=0)
            pbar.close()
            return x_all

    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(epoch):
        model.train()
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size
            pbar.update(batch.batch_size)
        pbar.close()

        return total_loss / total_examples, total_correct / total_examples

    @torch.no_grad()
    def test():
        model.eval()
        y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
        y = data.y.to(y_hat.device)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
        return accs

    for epoch in range(1, 11):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')

def load_social(dataset):
    edge_file = open(r"data/{}.edge".format(dataset), 'r')
    attri_file = open(r"data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print(
        "dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num,
                                                                      edge_num,
                                                                      attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)),
                        shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    features = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)),
                              shape=(node_num, attribute_number))

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                        shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj)
    fea_train, train_feas, val_feas, val_feas_false, \
    test_feas, test_feas_false = mask_test_feas(features)

    adj = adj_train
    features_orig = features
    features = sp.lil_matrix(features)

    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    return adj, features


def mask_test_edges(adj):
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333:
            if len(test_edges_false) < num_test:
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val:
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val:
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test:
                    test_edges_false.append([i, j])

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                              shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_feas(features):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 10.))
    num_val = int(np.floor(len(feas) / 20.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333:
            if len(test_feas_false) < num_test:
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val:
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val:
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test:
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])),
                              shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == '__main__':
    root = '/vol/research/sceneEvolution/data/graph_data'
    load_data(root, 'Flickr')