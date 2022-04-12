import matplotlib.pyplot as plt
import torch
from torch_cluster import fps
import pytorch3d
from pytorch3d.ops import sample_farthest_points
import torch.nn as nn
import torch.nn.functional as F
import matplotlib


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, self_loops_noise=False):
    """Draw a sample from the Gumbel-Softmax distribution
    Args:
        logits: input log probabilities of size [N, N]
    """

    if self_loops_noise:
        # Whether to add noise along diagonal of logits (self loop indices)
        noise = sample_gumbel(logits.size())
    else:
        # If no noise added to self loops then zero out those indices
        zero_self_loops = 1 - torch.eye(n=logits.shape[0])

        if torch.cuda.is_available():
            zero_self_loops = zero_self_loops.cuda()

        noise = sample_gumbel(logits.size()) * zero_self_loops

    y = logits + noise
    # y = F.softmax(y / temperature, dim=-1)
    return y


def gumbel_sample(logits, self_loops_noise=False):
    """Perturb sample with gumbel noise
    Args:
        logits: input log probabilities of size [N, N]
    """

    if self_loops_noise:
        # Whether to add noise across all logits including self loop indices (diagonal)
        noise = sample_gumbel(logits.size())
    else:
        # If no noise added to self loops then zero out those indices
        zero_self_loops = 1 - torch.eye(n=logits.shape[0])

        if torch.cuda.is_available():
            zero_self_loops = zero_self_loops.cuda()

        noise = sample_gumbel(logits.size()) * zero_self_loops

    y = logits + noise
    return y


def gumbel_perturb(logits, self_loops_noise=False, noise=True):
    """
      Sample from the Gumbel-Softmax distribution
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      self_loops_noise: if True, add noise to self-loop indices
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    """

    if noise:
        # During training add noise
        y = gumbel_sample(logits, self_loops_noise)
    else:
        # During inference no noise
        y = y

    return y


def straight_through_gumbel_softmax(
    logits, temperature, hard=False, self_loops_noise=False
):
    """
      Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, self_loops_noise)

    if not hard:
        # return y.view(-1, latent_dim * categorical_dim)
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    # return y_hard.view(-1, latent_dim * categorical_dim)
    return y_hard


def straight_through_gumbel_softmax_top_k(
    logits, temperature, k, hard=False, self_loops_noise=False, noise=True
):
    """
      Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
      self_loops_noise: if True, add noise to self-loop indices
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    if noise:
        # During training add noise
        y = gumbel_softmax_sample(logits, temperature, self_loops_noise)
    else:
        # During inference no noise
        y = F.softmax(logits / temperature, dim=-1)

    top_k_v, top_k_i = torch.topk(y, k=k, dim=-1, largest=True, sorted=False)

    if not hard:
        # return y.view(-1, latent_dim * categorical_dim)
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, top_k_i, 1)
    y_hard = y_hard.view(*shape)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y

    return y_hard


class DGG_StraightThrough(nn.Module):
    def __init__(
        self,
        in_dim=32,
        latent_dim=64,
        k=3,
        hard=True,
        self_loops_noise=False,
        dist_fn="mlp",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.k = k
        self.hard = hard
        self.self_loops_noise = self_loops_noise
        self.dist_fn = dist_fn

        # Embedding layers
        self.project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1),
        )

        # Distance function
        if dist_fn == "metric":
            self.t = nn.Parameter(torch.ones(1))
            nn.init.ones_(self.t)
        elif dist_fn == "mlp":
            self.distance = nn.Sequential(
                nn.Linear(latent_dim * 2, 1),
                nn.LeakyReLU(),
                nn.Softmax(dim=-1),
            )

    def forward(self, x, temp, noise=True):
        """

        Args:
            x: input points [B, N, dim]
        Returns:
            adj: adjacency between points in each batch [B, N, N]
        """

        # Embed input features [B, N, dim]
        x_proj = self.project(x)

        if self.dist_fn == "metric":
            # calculate distance/similarity between input nodes
            dist = torch.cdist(x, x, p=2)
            prob = torch.exp(-self.t * dist)  # [B, N, N]

        elif self.dist_fn == "mlp":
            N = x.shape[-2]

            # Pass points pairwise through MLP [B, N, dim] ---> [B, N, N, dim * 2]
            x_pairwise = torch.cat(
                [
                    x_proj.unsqueeze(2).repeat(1, 1, N, 1),
                    x_proj.unsqueeze(1).repeat(1, N, 1, 1),
                ],
                dim=-1,
            )

            prob = self.distance(x_pairwise).squeeze(-1)

        # Log probabilities
        log_p = torch.log(prob)

        # Gumbel softmax to get adjacency matrix    TODO: vectorise
        adj = [
            straight_through_gumbel_softmax_top_k(
                p, temp, self.k, self.hard, self.self_loops_noise, noise
            )
            for p in log_p
        ]

        return torch.stack(adj)


class DGG_LearnableK_SDD(nn.Module):
    """
    DGG_learnableK with x_support calculated in each forward pass to account
    for varying number of input nodes (N)
    """

    def __init__(
        self,
        in_dim=32,
        latent_dim=64,
        k_bias=1.0,
        hard=False,
        self_loops_noise=False,
        dist_fn="mlp",
        k_net_input="raw",
        hs_start=2,
        hs_end=-5,
        n_agents=None,
        learn_k_bias=None,
    ):
        super(DGG_LearnableK_SDD, self).__init__()

        torch.manual_seed(0)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hard = hard
        self.self_loops_noise = self_loops_noise
        self.dist_fn = dist_fn
        self.k_net_input = k_net_input

        # Embedding layers
        self.input_project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1),
        )

        # Distance function
        if dist_fn == "metric":
            self.t = nn.Parameter(torch.ones(1))
            nn.init.ones_(self.t)
        elif dist_fn == "mlp":
            self.distance = nn.Sequential(
                nn.Linear(latent_dim * 2, 1),
                nn.LeakyReLU(),
                nn.Softmax(dim=-1),
            )

        # Learnable K
        hs_start = (
            hs_start  # build x-axis points that will be projected onto smooth heaviside
        )
        hs_end = hs_end
        interval = hs_start - hs_end

        self.register_buffer("interval", torch.tensor(interval))
        self.register_buffer("k_bias", torch.tensor(k_bias))
        self.register_buffer("hs_start", torch.tensor(hs_start))
        self.register_buffer("hs_end", torch.tensor(hs_end))

        if k_net_input == "raw":
            # Option 1, use input to get mu, var in latent dim and then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=in_dim,
                latent_dim=latent_dim,
            )
        elif k_net_input == "embedded":
            # Option 3, use projected input to get mu, var in latent dim and
            # then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=latent_dim, latent_dim=latent_dim, option=1
            )

    def forward(self, x, temp, noise=True):
        """

        Args:
            x: input points [B, N, dim]
        Returns:
            adj: adjacency between points in each batch [B, N, N]
        """
        # get number of nodes
        N = x.shape[-2]

        # Embed input features [B, N, dim]
        x_proj = self.input_project(x)

        if self.dist_fn == "metric":
            # calculate distance/similarity between input nodes
            dist = torch.cdist(x_proj, x_proj, p=2)
            prob = torch.exp(-self.t * dist)  # [B, N, N]
        elif self.dist_fn == "mlp":
            # Pass points pairwise through MLP [B, N, dim] ---> [B, N, N, dim * 2]

            x_pairwise = torch.cat(
                [
                    x_proj.unsqueeze(2).repeat(1, 1, N, 1),
                    x_proj.unsqueeze(1).repeat(1, N, 1, 1),
                ],
                dim=-1,
            )
            prob = self.distance(x_pairwise).squeeze(-1)  # [B, N, N]

        # Log probabilities [B, N, N]
        log_p = torch.log(prob)

        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_prob = torch.stack(
                [
                    gumbel_softmax_sample(batch, temp, self.self_loops_noise)
                    for batch in log_p
                ]
            )
        else:
            edge_prob = F.softmax(log_p / temp, dim=-1)

        # Sort edge probabilities
        sorted, idxs = torch.sort(edge_prob, dim=-1, descending=True)

        # Get smooth top-k (smooth first-k elements really, as edge probs are sorted)
        if self.k_net_input == "raw":
            # use input to get k
            k = self.k_net(x)  # [B, N, 1]
        elif self.k_net_input == "embedded":
            # use projected input to get k
            k = self.k_net(x_proj)  # [B, N, 1]

        # Add bias to K, so it is always a minimum of the bias
        k = k + self.k_bias  # [B, N, 1]

        # Calculate first_k
        x_support = torch.arange(
            self.hs_start,
            -self.interval * (N - 1),
            step=-self.interval,
            device=k.device,
        )
        shift = -(k - 1) * -self.interval  # [B, N, 1]
        shift_support = (
            x_support + shift
        )  # TODO: normalise/scale so this doesnt blow up
        # first_k = smooth_heaviside(shift_support, t=0.0005)    # [B, N, N]
        first_k = torch.sigmoid(shift_support)  # TODO: replace with smooth_heaviside

        # Multiply sorted edge probabilities by first-k
        first_k_prob = sorted * first_k

        # Unsort
        adj = first_k_prob.clone().scatter_(dim=-1, index=idxs, src=first_k_prob)

        # check adjacency argmax equals edge_prob argmax
        # print(
        #     torch.all(torch.argmax(adj, dim=-1) == torch.argmax(edge_prob, dim=-1))
        # )

        if not self.hard:
            # return adjacency matrix with softmax probabilities
            return adj, k

        # if hard adj desired, get matrix of ones and multiply by first_k
        adj_hard = torch.ones_like(adj)
        adj_hard.scatter_(dim=-1, index=idxs, src=first_k)
        adj_hard = (adj_hard - adj).detach() + adj

        assert torch.any(torch.isnan(adj_hard)) == False
        assert torch.any(torch.isinf(adj_hard)) == False

        return adj_hard, k


class DGG_LearnableK_Small(nn.Module):
    """
    Differentiable graph generator with fixed x_support
    for use cases where number of input nodes is small
    """

    def __init__(
        self,
        in_dim=32,
        latent_dim=64,
        k_bias=1.0,
        hard=True,
        self_loops_noise=False,
        dist_fn="mlp",
        n_agents=1024,
        k_net_input="raw",
        hs_start=2,
        hs_end=-5,
        learn_k_bias=False,
    ):
        super(DGG_LearnableK, self).__init__()

        torch.manual_seed(0)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hard = hard
        self.self_loops_noise = self_loops_noise
        self.dist_fn = dist_fn
        self.k_net_input = k_net_input
        self.learn_k_bias = learn_k_bias

        # Embedding layers
        self.input_project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1),
        )

        # Distance function
        if dist_fn == "metric":
            self.t = nn.Parameter(torch.ones(1))
            nn.init.ones_(self.t)
        elif dist_fn == "mlp":
            self.distance = nn.Sequential(
                nn.Linear(latent_dim * 2, 1),
                nn.LeakyReLU(),
                nn.Softmax(dim=-1),
            )

        # Learnable K
        self.hs_start = (
            hs_start  # build x-axis points that will be projected onto smooth heaviside
        )
        self.hs_end = hs_end
        interval = self.hs_start - self.hs_end
        self.x_support = torch.arange(
            self.hs_start, -interval * (n_agents - 1), step=-interval
        ).cuda()

        self.register_buffer("interval", torch.tensor(interval))
        self.register_buffer("k_bias", torch.tensor(k_bias))

        if k_net_input == "raw":
            # Option 1, use input to get mu, var in latent dim and then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=in_dim, latent_dim=latent_dim, learn_k_bias=learn_k_bias
            )
        elif k_net_input == "embedded":
            # Option 3, use projected input to get mu, var in latent dim and
            # then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=latent_dim, latent_dim=latent_dim, learn_k_bias=learn_k_bias
            )
        elif k_net_input == "edge_prob":
            # Option 4, use edge probabilities to get mu, var in latent dim
            # and then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=n_agents, latent_dim=latent_dim, learn_k_bias=learn_k_bias
            )

    def forward(self, x, temp, noise=True):
        """
        Args:
            x: input points [B, N, dim]
        Returns:
            adj: adjacency between points in each batch [B, N, N]
        """

        # Embed input features [B, N, dim]
        x_proj = self.input_project(x)

        if self.dist_fn == "metric":
            # calculate distance/similarity between input nodes
            dist = torch.cdist(x_proj, x_proj, p=2)
            prob = torch.exp(-self.t * dist)  # [B, N, N]
        elif self.dist_fn == "mlp":
            # Pass points pairwise through MLP [B, N, dim] ---> [B, N, N, dim * 2]
            N = x.shape[-2]
            x_pairwise = torch.cat(
                [
                    x_proj.unsqueeze(2).repeat(1, 1, N, 1),
                    x_proj.unsqueeze(1).repeat(1, N, 1, 1),
                ],
                dim=-1,
            )
            prob = self.distance(x_pairwise).squeeze(-1)  # [B, N, N]

        # Log probabilities [B, N, N]
        log_p = torch.log(prob)

        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_prob = torch.stack(
                [
                    gumbel_softmax_sample(batch, temp, self.self_loops_noise)
                    for batch in log_p
                ]
            )
        else:
            edge_prob = F.softmax(log_p / temp, dim=-1)

        # Sort edge probabilities
        sorted, idxs = torch.sort(edge_prob, dim=-1, descending=True)

        # Get smooth top-k (smooth first-k elements really, as edge probs are sorted)
        if self.k_net_input == "raw":
            # use input to get k
            k = self.k_net(x)
        elif self.k_net_input == "embedded":
            # use projected input to get k
            k = self.k_net(x_proj)
        elif self.k_net_input == "edge_prob":
            # use perturbed edge probabilities to get k
            k = self.k_net(edge_prob)

        # if not learning the bias, then just add a constant
        if not self.learn_k_bias:
            # Add bias to K, so it is always a minimum of the bias
            k = k + self.k_bias

        shift = -(k - 1) * -self.interval
        shift_support = self.x_support + shift
        first_k = smooth_heaviside(shift_support, t=0.5)  # [B, N, N]

        # Multiply sorted edge probabilities by first-k
        first_k_prob = sorted * first_k

        # Unsort
        adj = first_k_prob.clone().scatter_(dim=-1, index=idxs, src=first_k_prob)

        # check adjacency argmax equals edge_prob argmax
        # print(
        #     torch.all(torch.argmax(adj, dim=-1) == torch.argmax(edge_prob, dim=-1))
        # )

        if not self.hard:
            # return adjacency matrix with softmax probabilities
            return adj, idxs

        # if hard adj desired, get matrix of ones and multiply by first_k
        adj_hard = torch.ones_like(adj)
        adj_hard.scatter_(dim=-1, index=idxs, src=first_k)
        adj_hard = (adj_hard - adj).detach() + adj
        return adj_hard, idxs


class DGG_LearnableK_old(nn.Module):
    """
    DGG_learnableK with x_support calculated in each forward pass to account
    for varying number of input nodes (N)

    Learnable k bit is from ECCV and is weird and its hyperparameters need
    to be tailored to the number of input points
    """

    def __init__(
        self,
        in_dim=32,
        latent_dim=64,
        k_bias=1.0,
        hard=False,
        self_loops_noise=False,
        dist_fn="mlp",
        k_net_input="raw",
        hs_start=2,
        hs_end=-5,
    ):
        super(DGG_LearnableK, self).__init__()

        torch.manual_seed(0)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hard = hard
        self.self_loops_noise = self_loops_noise
        self.dist_fn = dist_fn
        self.k_net_input = k_net_input

        # Embedding layers
        self.input_project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1),
        )

        # Distance function
        if dist_fn == "metric":
            self.t = nn.Parameter(torch.ones(1))
            nn.init.ones_(self.t)
        elif dist_fn == "mlp":
            self.distance = nn.Sequential(
                nn.Linear(latent_dim * 2, 1),
                nn.LeakyReLU(),
                nn.Softmax(dim=-1),
            )

        # Learnable K
        hs_start = (
            hs_start  # build x-axis points that will be projected onto smooth heaviside
        )
        hs_end = hs_end
        interval = hs_start - hs_end

        self.register_buffer("interval", torch.tensor(interval))
        self.register_buffer("k_bias", torch.tensor(k_bias))
        self.register_buffer("hs_start", torch.tensor(hs_start))
        self.register_buffer("hs_end", torch.tensor(hs_end))

        if k_net_input == "raw":
            # Option 1, use input to get mu, var in latent dim and then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=in_dim,
                latent_dim=latent_dim,
            )
        elif k_net_input == "embedded":
            # Option 3, use projected input to get mu, var in latent dim and
            # then project down to 1
            self.k_net = LearnableKEncoder(
                in_dim=latent_dim, latent_dim=latent_dim, option=1
            )

    def forward(self, x, temp, noise=True):
        """

        Args:
            x: input points [B, N, dim]
        Returns:
            adj: adjacency between points in each batch [B, N, N]
        """
        # get number of nodes
        N = x.shape[-2]

        # Embed input features [B, N, dim]
        x_proj = self.input_project(x)

        if self.dist_fn == "metric":
            # calculate distance/similarity between input nodes
            dist = torch.cdist(x_proj, x_proj, p=2)
            prob = torch.exp(-self.t * dist)  # [B, N, N]
        elif self.dist_fn == "mlp":
            # Pass points pairwise through MLP [B, N, dim] ---> [B, N, N, dim * 2]

            x_pairwise = torch.cat(
                [
                    x_proj.unsqueeze(2).repeat(1, 1, N, 1),
                    x_proj.unsqueeze(1).repeat(1, N, 1, 1),
                ],
                dim=-1,
            )
            prob = self.distance(x_pairwise).squeeze(-1)  # [B, N, N]

        # Log probabilities [B, N, N]
        log_p = torch.log(prob)

        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_prob = torch.stack(
                [
                    gumbel_softmax_sample(batch, temp, self.self_loops_noise)
                    for batch in log_p
                ]
            )
        else:
            edge_prob = F.softmax(log_p / temp, dim=-1)

        # Sort edge probabilities
        sorted, idxs = torch.sort(edge_prob, dim=-1, descending=True)

        # Get smooth top-k (smooth first-k elements really, as edge probs are sorted)
        if self.k_net_input == "raw":
            # use input to get k
            k = self.k_net(x)  # [B, N, 1]
        elif self.k_net_input == "embedded":
            # use projected input to get k
            k = self.k_net(x_proj)  # [B, N, 1]

        # Add bias to K, so it is always a minimum of the bias
        k = k + self.k_bias  # [B, N, 1]

        # Calculate first_k
        x_support = torch.arange(
            self.hs_start,
            -self.interval * (N - 1),
            step=-self.interval,
            device=k.device,
        )
        shift = -(k - 1) * -self.interval  # [B, N, 1]
        shift_support = (
            x_support + shift
        )  # TODO: normalise/scale so this doesnt blow up
        # first_k = smooth_heaviside(shift_support, t=0.0005)    # [B, N, N]
        first_k = torch.sigmoid(shift_support)  # TODO: replace with smooth_heaviside

        # Multiply sorted edge probabilities by first-k
        first_k_prob = sorted * first_k

        # Unsort
        adj = first_k_prob.clone().scatter_(dim=-1, index=idxs, src=first_k_prob)

        # check adjacency argmax equals edge_prob argmax
        # print(
        #     torch.all(torch.argmax(adj, dim=-1) == torch.argmax(edge_prob, dim=-1))
        # )

        if not self.hard:
            # return adjacency matrix with softmax probabilities
            return adj

        # if hard adj desired, get matrix of ones and multiply by first_k
        adj_hard = torch.ones_like(adj)
        adj_hard.scatter_(dim=-1, index=idxs, src=first_k)
        adj_hard = (adj_hard - adj).detach() + adj

        assert torch.any(torch.isnan(adj_hard)) == False
        assert torch.any(torch.isinf(adj_hard)) == False

        return adj_hard, idxs


class DGG_LearnableK(nn.Module):
    """
    DGG_learnableK with x_support calculated in each forward pass to account
    for varying number of input nodes (N)
    """

    def __init__(
        self,
        in_dim=32,
        adj_dim=1,
        latent_dim=64,
        k_bias=1.0,
        hard=False,
        self_loops_noise=False,
        dist_fn="mlp",
        k_net_input="raw",
        hs_start=2,
        hs_end=-5,
    ):
        super(DGG_LearnableK, self).__init__()

        # torch.manual_seed(0)

        self.in_dim = in_dim
        self.adj_dim = adj_dim
        self.latent_dim = latent_dim
        self.hard = hard
        self.self_loops_noise = self_loops_noise
        self.dist_fn = dist_fn
        self.k_net_input = k_net_input

        # Embedding layers
        self.input_project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self.input_degree_project = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.LeakyReLU(),
        )
        self.combine_input_degree = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LeakyReLU(),
        )

        self.input_adj_project = nn.Sequential(
            nn.Linear(in_dim * 2 + adj_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
        )

        self.adj_project = nn.Sequential(
            nn.Linear(adj_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )

        # Learnable K
        self.register_buffer("k_bias", torch.tensor(k_bias))


        # Option 3, use projected input to get mu, var in latent dim and
        # then project down to 1
        self.k_net = LearnableKEncoder(
            in_dim=latent_dim, latent_dim=latent_dim
        )

    def forward(self, x, in_adj, temp, noise=True, writer=None, epoch=None):
        """

        Args:
            x: input points [N, dim]
            in_adj: unnormalized sparse adjacency matrix (coalesced) [N, N]
        Returns:
            adj: adjacency matrix [N, N]
        """
        assert x.ndim ==  2
        assert len(in_adj.shape) == 2

        # get number of nodes
        N = x.shape[-2]

        # embed input and adjacency to get initial edge log probabilities
        edge_p = self.edge_prob_net(in_adj, x, bypass='project_adj')               # [N, N]
        return edge_p.to_dense().unsqueeze(0)  # step 0

        log_p = torch.sparse.log_softmax(edge_p, dim=1)  # [N, N]

        # prepare input features and log probs for rest of function
        x = x.unsqueeze(0)                          # [1, N, dim]
        log_p = log_p.to_dense().unsqueeze(0)       # [1, N, N]

        # add gumbel noise to edge log probabilities
        log_p = self.perturb_edge_prob(log_p, noise, temp)

        # Sort edge probabilities in ASCENDING order
        sorted_log_p, idxs = torch.sort(log_p, dim=-1, descending=False)

        # Get smooth top-k
        k = self.k_estimate_net(N, in_adj, x)    # [1, N, 1]

        # select top_k
        adj, top_k = self.select_top_k(N, idxs, k, sorted_log_p, temp)

        if not self.hard:
            # return adjacency matrix with softmax probabilities
            return adj, k

        # if hard adj desired, threshold first_k and scatter
        adj_hard = torch.ones_like(adj)
        adj_hard.scatter_(dim=-1, index=idxs, src=(top_k > 0.8).float())
        adj_hard = (adj_hard - adj).detach() + adj

        assert torch.any(torch.isnan(adj_hard)) == False
        assert torch.any(torch.isinf(adj_hard)) == False

        # plt.imshow(edge_prob.detach().cpu()[0])
        # plt.title('edge prob')
        # plt.show()
        # plt.imshow(sorted.detach().cpu()[0])
        # plt.title('sorted')
        # plt.show()
        # plt.imshow(first_k.detach().cpu()[0])
        # plt.title('first k')
        # plt.show()
        # plt.imshow(first_k_prob.detach().cpu()[0])
        # plt.title('first k prob')
        # plt.show()
        # plt.imshow(adj.detach().cpu()[0])
        # plt.title('adj')
        # plt.show()
        # plt.imshow(adj_hard.detach().cpu()[0])
        # plt.title('adj hard')
        # plt.show()

        return adj_hard, k

    def select_top_k(self, N, idxs, k, sorted_log_p, temp):
        t = torch.arange(N).reshape(1, 1, -1).cuda()  # base domain
        t = (t / N) * 2 - 1  # squeeze to [-1, 1]
        w = 0.001  # sharpness parameter
        first_k = (1 + torch.tanh((t + k) / w))  # higher k = more items closer to 1
        # print('    first k', first_k)
        # Multiply sorted edge log probabilities by first-k and then softmax
        first_k_log_prob = sorted_log_p * first_k
        first_k_prob = torch.softmax(first_k_log_prob / temp, dim=-1)
        # print('    first k prob', first_k_prob)
        # Unsort
        adj = first_k_prob.clone().scatter_(dim=-1, index=idxs, src=first_k_prob)
        return adj, first_k

    def k_estimate_net(self, N, in_adj, x):
        in_degree = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
        # k = (in_degree / N) * 2 - 1
        degree = self.input_degree_project(in_degree)  # [1, N, dim]
        x_proj = self.input_project(x)  # [1, N, dim]
        feats_for_k = torch.cat([degree, x_proj], dim=-1)  # [1, N, 2 x dim]
        in_k_feats = self.combine_input_degree(feats_for_k)

        # use projected input to get k
        k = self.k_net(in_k_feats)  # [B, N, 1]
        # # Keep k between -1 and 1
        # k = torch.tanh(k)
        return k

    def perturb_edge_prob(self, log_p, noise, temp):
        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_log_p = torch.stack(
                [
                    gumbel_softmax_sample(batch, temp, self.self_loops_noise)
                    for batch in log_p
                ]
            )
        else:
            edge_log_p = log_p
        return edge_log_p

    def edge_prob_net(self, in_adj, x, bypass=None):
        if bypass is None:
            u = x[in_adj.indices()[0, :]]  # [n, dim]
            v = x[in_adj.indices()[1, :]]  # [n, dim]
            auv = in_adj.values().unsqueeze(-1)  # [n, 1]
            u_v_auv = torch.concat([u, v, auv], dim=-1)  # [n, dim + dim + 1]
            z = self.input_adj_project(u_v_auv).flatten()  # [n]
            z_matrix = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z_matrix
        elif bypass == 'pass':
            # for debugging purposes
            z_matrix = torch.sparse.FloatTensor(
                in_adj.indices(), in_adj.values(), in_adj.shape
            )
            return z_matrix
        elif bypass == 'project_adj':
            auv = in_adj.values().unsqueeze(-1)  # [n, 1]
            z = self.adj_project(auv).flatten()  # [n]
            z_matrix = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z_matrix


class LearnableKEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, learn_k_bias=False):
        super(LearnableKEncoder, self).__init__()

        self.learn_k_bias = learn_k_bias
        # Option 1, use input to get mu, var in latent dim and
        # then project down to 1
        self.k_mu = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        self.k_logvar = nn.Linear(in_dim, latent_dim)
        self.k_project = nn.Sequential(
            nn.Linear(latent_dim, 1),
            # nn.ReLU(inplace=True)  # want k to be positive
        )

    def latent_sample(self, mu, logvar):
        if self.training:  # TODO: change from inplace operations to normal
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            # return mu
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)

    def forward(self, x):
        latent_k_mu = self.k_mu(x)
        # latent_k_logvar = self.k_logvar(x)
        # latent_k = self.latent_sample(latent_k_mu, latent_k_logvar)
        k = self.k_project(latent_k_mu)  # [B, N, 1]

        return k


def smooth_heaviside(x, t):
    return 1 / (1 + torch.exp(-2 * t * x))


def sigmoid(x, shift):
    return 1 / (1 + torch.exp(-(x - shift)))


def fps(x, K):
    """
    farthest point sampling
    """
    # index = fps(x, ratio=ratio, random_start=random_start)
    points, idxs = pytorch3d.ops.sample_farthest_points(points=x, K=K)
    return points, idxs


### ------- debug functions ------- ###
def graph_sampler_top_k(k, temp, hard=True, diff_adj=True):
    torch.manual_seed(0)
    N = 5
    dim = 5
    x = torch.rand(N, dim, requires_grad=True).cuda()
    x.retain_grad()
    prob = torch.softmax(x, dim=-1)
    prob.retain_grad()
    print("prob", get_tensor_attributes(prob))
    print(prob)

    # Log probabilities [N, N]
    log_p = torch.log(prob)
    log_p.retain_grad()
    print("log p", get_tensor_attributes(log_p))
    print(log_p)

    # Gumbel softmax to get adjacency matrix
    adj = straight_through_gumbel_softmax_top_k(log_p, temp, k=k, hard=True)
    adj.retain_grad()
    assert adj.shape == log_p.shape
    print("adj", get_tensor_attributes(adj))
    print(adj)

    # GAT
    in_feat = dim
    out_feat = 1
    if diff_adj:
        gat = GraphAttentionLayerDiffAdj(
            in_features=in_feat, out_features=out_feat, alpha=0.2
        )
        gat.to("cuda")
    else:
        gat = GraphAttentionLayer(in_features=in_feat, out_features=out_feat, alpha=0.2)
        gat.to("cuda")
    y = gat(x, adj)
    y.retain_grad()
    print("y", get_tensor_attributes(y))
    print(y)

    gt = torch.ones_like(y, requires_grad=False)

    loss = ((gt - y) ** 2).sum()
    # loss = (adj ** 2).sum()
    loss.backward()

    print("y grad")
    print(y.grad)

    print("adj grad")
    print(adj.grad)

    print("log p")
    print(log_p.grad)

    print("prob grad")
    print(prob.grad)

    print("x grad")
    print(x.grad)
    return


def DGM(k, temp, hard=True, diff_adj=True, self_loops_noise=False):
    torch.manual_seed(0)
    N = 5
    dim = 2

    dgm = DGG_StraightThrough(
        in_dim=dim, latent_dim=5, k=k, hard=hard, self_loops_noise=self_loops_noise
    )
    dgm.to("cuda")

    X = torch.rand(1, N, dim, requires_grad=True).cuda()
    X.retain_grad()
    print("x", get_tensor_attributes(X))
    print(X)

    # Gumbel softmax to get adjacency matrix
    adj = dgm(X, temp)
    adj.retain_grad()
    print("adj", get_tensor_attributes(adj))
    print(adj)

    # GAT
    in_feat = dim
    out_feat = 1
    if diff_adj:
        gat = GraphAttentionLayerDiffAdj(
            in_features=in_feat, out_features=out_feat, alpha=0.2
        )
        gat.to("cuda")
    else:
        gat = GraphAttentionLayer(in_features=in_feat, out_features=out_feat, alpha=0.2)
        gat.to("cuda")
    y = torch.stack([gat(x, a) for x, a in zip(X, adj)])
    y.retain_grad()
    print("y", get_tensor_attributes(y))
    print(y)

    gt = torch.ones_like(y, requires_grad=False)

    loss = ((gt - y) ** 2).sum()
    # loss = (adj ** 2).sum()
    loss.backward()

    print("y grad")
    print(y.grad)

    print("adj grad")
    print(adj.grad)

    print("x grad")
    print(X.grad)

    return


def graph_sampler(temp, hard=True, diff_adj=True):
    """ """

    torch.manual_seed(0)
    N = 5
    dim = 5
    x = torch.rand(N, dim, requires_grad=True)
    prob = torch.softmax(x, dim=-1)
    prob.retain_grad()
    print("prob", get_tensor_attributes(prob))
    print(prob)

    # Log probabilities [N, N]
    log_p = torch.log(prob)
    log_p.retain_grad()
    print("log p", get_tensor_attributes(log_p))
    print(log_p)

    # Gumbel softmax to get adjacency matrix
    adj = straight_through_gumbel_softmax(log_p, temp, hard=True)
    adj.retain_grad()
    assert adj.shape == log_p.shape
    print("adj", get_tensor_attributes(adj))
    print(adj)

    # GAT
    in_feat = dim
    out_feat = 1
    if diff_adj:
        gat = GraphAttentionLayerDiffAdj(
            in_features=in_feat, out_features=out_feat, alpha=0.2
        )
    else:
        gat = GraphAttentionLayer(in_features=in_feat, out_features=out_feat, alpha=0.2)
    y = gat(x, adj)
    y.retain_grad()
    print("y", get_tensor_attributes(y))
    print(y)

    gt = torch.ones_like(y, requires_grad=False)

    loss = ((gt - y) ** 2).sum()
    # loss = (adj ** 2).sum()
    loss.backward()

    print("y grad")
    print(y.grad)

    print("adj grad")
    print(adj.grad)

    print("log p")
    print(log_p.grad)

    print("prob grad")
    print(prob.grad)

    print("x grad")
    print(x.grad)
    return


def graph_conv():
    # Input node features [N, 2]
    torch.manual_seed(0)
    N = 3
    x = torch.rand(N, 2, requires_grad=True)
    print("x")
    print(x)
    # Edge probabilities
    p = x @ x.T
    print("p")
    print(p)
    # Gumbel softmax to get adjacency matrix
    temp = 0.1
    adj = straight_through_gumbel_softmax(p, temp, hard=True)
    print("adj")
    print(adj)
    # Graph conv
    W = torch.rand(1, 2, requires_grad=True)
    h = x.T @ adj
    print("h")
    print(h)
    y = W @ h
    print("gconv", y.shape)
    print(y)
    loss = (adj ** 2).sum()
    loss.backward()
    print("w grad")
    print(W.grad)
    print("adj grad")
    print(adj.grad)
    print("p grad")
    print(p.grad)
    print("x grad")
    print(x.grad)


def gumbel_softmax_diff_test(hard=False):
    # Input probabilities [N, 2]
    torch.manual_seed(0)
    N = 5
    dim = 2
    x = torch.log(torch.rand(N, dim, requires_grad=True))
    x.retain_grad()
    print("x")
    print(x)

    # Gumbel softmax to get z
    temp = 0.1
    z = straight_through_gumbel_softmax(x, temp, hard=hard)
    z.retain_grad()
    print("z")
    print(z)

    # loss
    gt = (torch.rand(N, dim, requires_grad=False) > 0.5).float().detach()
    loss = ((gt - z) ** 2).sum()
    loss.backward()

    print("gt")
    print(gt)

    print("z grad")
    print(z.grad)

    print("x grad")
    print(x.grad)


def gumbel_softmax_top_k_diff_test(hard=False):
    # Input probabilities [N, 2]
    torch.manual_seed(0)
    N = 5
    dim = 2
    x = torch.log(torch.rand(N, dim, requires_grad=True))
    x.retain_grad()
    print("x")
    print(x)

    # Gumbel softmax to get z
    temp = 0.1
    z = straight_through_gumbel_softmax_top_k(x, temp, k=1, hard=hard)
    z.retain_grad()
    print("z")
    print(z)

    # loss
    gt = (torch.rand(N, dim, requires_grad=False) > 0.5).float().detach()
    loss = ((gt - z) ** 2).sum()
    loss.backward()

    print("gt")
    print(gt)

    print("z grad")
    print(z.grad)

    print("x grad")
    print(x.grad)


def smooth_heaviside_test():
    global interval
    x = torch.arange(-15, 15, 0.25)
    y = smooth_heaviside(x, t=0.5)
    plt.plot(x, y, ms=30)
    plt.grid()
    N = 5
    x_at_y1 = 2
    x_at_y0 = -5
    interval = x_at_y1 - x_at_y0
    x_support = torch.arange(x_at_y1, -interval * N, step=-interval)
    print(x_support)
    k = 1
    shift = -(k - 1) * -interval
    x_support = x_support + shift
    y_x_support = smooth_heaviside(x_support, t=0.5)
    plt.scatter(x_support, y_x_support, s=300)
    print("k 1", y_x_support)
    k = 2
    shift = -(k - 1) * -interval
    x_support = x_support + shift
    y_x_support = smooth_heaviside(x_support, t=0.5)
    plt.scatter(x_support, y_x_support, s=150)
    print("k 2", y_x_support)
    #
    k = 2.5
    shift = -(k - 1) * -interval
    x_support = x_support + shift
    y_x_support = smooth_heaviside(x_support, t=0.5)
    plt.scatter(x_support, y_x_support, s=50)
    print("k 2.5", y_x_support)
    plt.show()


def diff_top_k_test():
    x = (
        torch.tensor([[0.0, 0], [0, 2], [0, 4], [2, 0], [2, 2]], requires_grad=True)
        .cuda()
        .unsqueeze(0)
    )
    w = torch.ones([2, 2]).cuda()
    print(x @ w)
    x.retain_grad()
    print("x")
    print(x)
    dgg = DGG_DiffTopK(
        in_dim=2, latent_dim=2, k=3, self_loops_noise=False, epsilon=0.0001
    ).cuda()
    adj = dgg(x, noise=True)
    print("adj")
    print(adj)
    adj.retain_grad()
    loss = (adj ** 2).sum()
    loss.backward()
    print(x.grad)
    print(adj.grad)
    x = torch.arange(-10, 10, 0.5) - 10
    y = smooth_heaviside(x, k=1)
    plt.scatter(x, y, s=30)


def learnable_k_test(input, hard):
    x = (
        torch.tensor([[0.0, 0], [0, 2], [0, 4], [2, 0], [2, 2]], requires_grad=True)
        .cuda()
        .unsqueeze(0)
    )
    w = torch.ones([2, 2]).cuda()
    print(x @ w)
    x.retain_grad()
    print("x")
    print(x)
    dgg = DGG_LearnableK(
        in_dim=2,
        latent_dim=2,
        k_bias=1,
        self_loops_noise=False,
        hard=hard,
        dist_fn="metric",
        n_agents=x.shape[-2],
        k_net_input=input,
    ).cuda()
    adj = dgg(x, temp=1, noise=True)

    print("adj")
    print(adj)
    adj.retain_grad()
    loss = (adj ** 2).sum()
    loss.backward()

    print(x.grad)
    print(adj.grad)
    print(dgg.k_net.k_mu.weight.grad)
    print(dgg.k_net.k_project[0].weight.grad)


def point_sampler():
    x = torch.arange(5)
    y = torch.arange(5)

    grid_x, grid_y = torch.meshgrid(x, y)
    x = torch.stack([grid_x, grid_y]).reshape(2, -1).T.unsqueeze(0)

    idx = fps(x, ratio=0.5, random_start=False)
    print(x)
    print(idx)

    plt.scatter(x[0, :, 0], x[0, :, 1], c="b")
    plt.scatter(idx[0, :, 0], idx[0, :, 1], c="r")
    plt.show()


def tanh_test_exp(k):
    N = 30
    k = torch.tensor(k, requires_grad=True)
    w = torch.tensor(1.0, requires_grad=True)
    t = torch.arange(start=-N, end=0.0)

    exp_k = torch.exp(k)
    y = 0.5 * (1 + torch.tanh((t + exp_k) / w))

    x = torch.ones_like(t, requires_grad=True)
    x_topk_gt = (torch.rand(x.shape) > 0.5).float()

    x_topk = x * y

    # loss = (torch.abs(x_topk - x_topk_gt) ** 2).sum()
    loss = (x_topk ** 2).sum()

    y.retain_grad()
    w.retain_grad()
    k.retain_grad()
    x.retain_grad()

    loss.backward()

    print("k", k.grad)
    print("x", x.grad)
    print("w", w.grad)
    print("y", y.grad)

    # plt.scatter(t, x_topk.detach())
    # # plt.scatter(t, y.detach())
    # plt.scatter(t, x_topk_gt.detach())
    # plt.show()


def tanh_test(k):
    N = 3000
    k = torch.tensor(k, requires_grad=True)
    w = torch.tensor(1.0, requires_grad=True)
    t = torch.arange(start=-N, end=0.0)

    y = 0.5 * (1 + torch.tanh((t + k) / w))

    x = torch.ones_like(t, requires_grad=True)
    x_topk_gt = (torch.rand(x.shape) > 0.5).float()

    x_topk = x * y

    # loss = (torch.abs(x_topk - x_topk_gt) ** 2).sum()
    loss = (x_topk ** 2).sum()

    y.retain_grad()
    w.retain_grad()
    k.retain_grad()
    x.retain_grad()

    loss.backward()

    print("k", k.grad)
    print("x", x.grad)
    print("w", w.grad)
    print("y", y.grad)

    # plt.scatter(t, x_topk.detach())
    # # plt.scatter(t, y.detach())
    # plt.scatter(t, x_topk_gt.detach())
    # plt.show()

def tanh_plot(k=0, w=1):
    N = 50
    x = torch.arange(N)
    x = (x / N) * 2 - 1
    y = torch.tanh((x + k) / w)

    plt.scatter(x, 0.5 * (1 + y))
    plt.scatter(x, torch.sigmoid(y))



if __name__ == "__main__":

    # tanh_test_exp(2.5)
    # tanh_test(12.2)

    tanh_plot(k=0, w=0.5)
    tanh_plot(k=0.5, w=0.5)
    # tanh_plot(k=-0.5, w=0.5)
    tanh_plot(k=-1, w=0.5)
    plt.show()

