import matplotlib.pyplot as plt
import torch
from torch_cluster import fps
# import pytorch3d
# from pytorch3d.ops import sample_farthest_points
import torch.nn as nn
import torch.nn.functional as F
import matplotlib


def sample_gumbel_from_uniform(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sample(logits, noise_sample):
    """Draw a sample from the Gumbel-Softmax distribution
    Args:
        logits: input log probabilities of size [N, N]
    """
    assert logits.shape == noise_sample.shape

    # No noise added to self loops so zero out those indices
    zero_self_loops = 1 - torch.eye(n=logits.shape[0])

    if torch.cuda.is_available():
        zero_self_loops = zero_self_loops.cuda()

    noise = noise_sample * zero_self_loops
    y = logits + noise_sample
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
    y = gumbel_sample(logits, temperature, self_loops_noise)

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
        y = gumbel_sample(logits, temperature, self_loops_noise)
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
                [gumbel_sample(batch, temp, self.self_loops_noise) for batch in log_p]
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
                [gumbel_sample(batch, temp, self.self_loops_noise) for batch in log_p]
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
                [gumbel_sample(batch, temp, self.self_loops_noise) for batch in log_p]
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
        degree_mean=3,
        degree_std=5,
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
        self.deg_mean = degree_mean
        self.deg_std = degree_std

        # Embedding layers
        self.input_project = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self.input_degree_project = nn.Linear(1, 3, bias=True)
        self.input_degree_decode = nn.Linear(3, 1, bias=True)
        self.combine_input_degree = nn.Sequential(
            nn.Linear(latent_dim + 3, latent_dim),
            nn.LeakyReLU(),
        )
        self.degree_bias = nn.Parameter(torch.ones(1) * -1)

        self.input_adj_project = nn.Sequential(
            nn.Linear(in_dim * 2 + adj_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.adj_project = nn.Sequential(
            nn.Linear(adj_dim, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
        )

        # Learnable K
        self.register_buffer("k_bias", torch.tensor(k_bias))

        # Option 3, use projected input to get mu, var in latent dim and
        # then project down to 1
        self.k_net = LearnableKEncoder(in_dim=3, latent_dim=3)

        self.gumbel = torch.distributions.Gumbel(
            loc=torch.tensor(0.0), scale=torch.tensor(0.3)
        )

    def k_hook(self, grad):
        grad = torch.clamp(grad, min=-0.05, max=0.05)
        self.k_grad.append(grad)

    def forward(self, x, in_adj, temp, noise=True, writer=None, epoch=None):
        """

        Args:
            x: input points [N, dim]
            in_adj: unnormalized sparse adjacency matrix (coalesced) [N, N]
        Returns:
            adj: adjacency matrix [N, N]
        """
        assert x.ndim == 2
        assert len(in_adj.shape) == 2

        # get number of nodes
        N = x.shape[-2]

        # prepare input featuresfor rest of function
        x = x.unsqueeze(0)  # [1, N, dim]

        # embed input and adjacency to get initial edge log probabilities
        edge_p_mode = None
        edge_p = self.edge_prob_net(in_adj, x, mode=edge_p_mode)  # [N, N]
        edge_p = edge_p.to_dense().unsqueeze(0)  # [1, N, N]
        edge_p = F.relu(edge_p)  # keep edge probabilities positive
        return edge_p  # STEP 0

        # add gumbel noise to edge log probabilities
        edge_p = edge_p + 1e-8
        log_p = torch.log(edge_p)  # [1, N, N]
        gumbel_noise = self.gumbel.sample(log_p.shape).cuda()

        noise_mode = "everywhere"
        if noise_mode == "non_edges":
            # only keep noise on non edges
            mask = 1 - in_adj.to_dense()
            gumbel_noise = gumbel_noise * mask
            gumbel_noise = torch.clamp(gumbel_noise, max=0.5)
        elif noise_mode == "positive":
            gumbel_noise = torch.clamp(gumbel_noise, min=0, max=1)
        elif noise_mode == "everywhere":
            pass

        pert_log_p = self.perturb_edge_prob(
            log_p, noise_sample=gumbel_noise, noise=True
        )
        pert_edge_p = torch.exp(pert_log_p)
        # return pert_edge_p   # STEP 1

        # get smooth top-k
        k, log_k = self.k_estimate_net(
            N, in_adj, x, mode="learn_normalized_degree_relu"
        )  # [1, N, 1]
        if writer is not None:
            writer.add_scalar("values/k_std", log_k.std(), epoch)
            writer.add_scalar("values/k_mean", log_k.mean(), epoch)

        # register hooks for gradients
        if self.training:
            self.k_grad = []
            k_grad = k.register_hook(self.k_hook)
            k.retain_grad()

        # select top_k
        topk_edge_p, top_k = self.select_top_k(
            N, k, pert_edge_p, mode="k_only", writer=writer, epoch=epoch
        )

        if writer is not None:
            writer.add_scalar("values/first_k_std", top_k.sum(-1).std(), epoch)
            writer.add_scalar("values/first_k_mean", top_k.sum(-1).mean(), epoch)

        if not self.hard:
            # return adjacency matrix with softmax probabilities
            return topk_edge_p

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

    def select_top_k(
        self, N, k, pert_edge_p, mode="k_times_edge_prob", writer=None, epoch=None
    ):
        if mode == "k_times_edge_prob":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            if writer is not None:
                writer.add_scalar("values/edge_p_std", s_edge_p.sum(-1).std(), epoch)
                writer.add_scalar("values/edge_p_mean", s_edge_p.sum(-1).mean(), epoch)

            t = torch.arange(N).reshape(1, 1, -1).cuda()  # base domain
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1

            # Multiply sorted edge log probabilities by first-k and then softmax
            first_k_log_p = s_edge_p * first_k

            # Unsort
            adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
            return adj, first_k

        elif mode == "k_only":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1

            # Unsort
            adj = first_k.clone().scatter_(dim=-1, index=idxs, src=first_k)
            return adj, first_k

        # k_only with linear gradients (instead of tanh saturating grads)
        elif mode == "k_only_w_linear_grad":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            first_k = -t + k

            # clamp values in forward to [0, 1] without affecting backward
            with torch.no_grad():
                first_k[:] = torch.clamp(first_k, min=0, max=1)

            # Unsort
            adj = first_k.clone().scatter_(dim=-1, index=idxs, src=first_k)
            return adj, first_k

        # k_only with linear gradients (instead of tanh saturating grads)
        elif mode == "k_times_edge_prob_w_linear_grad":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            w = 1  # sharpness parameter
            first_k = -t + k

            # Multiply sorted edge log probabilities by first-k and then softmax
            first_k_log_p = s_edge_p * first_k

            # clamp values in forward to [0, 1] without affecting backward
            with torch.no_grad():
                first_k_log_p[:] = torch.clamp(first_k_log_p, min=0, max=1)

            # Unsort
            adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
            return adj, first_k_log_p

    def k_estimate_net(self, N, in_adj, x, mode="calculate"):
        if mode == "calculate":
            in_degree = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            k = (in_degree / N) * 2 - 1
        elif mode == "learn":
            in_degree = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            degree = F.leaky_relu(self.input_degree_project(in_degree))  # [1, N, dim]

            x_proj = self.input_project(x)  # [1, N, dim]
            feats_for_k = torch.cat([degree, x_proj], dim=-1)  # [1, N, 2 x dim]

            in_k_feats = self.combine_input_degree(feats_for_k)

            # use projected input to get k
            log_k = self.k_net(in_k_feats)  # [B, N, 1]
            og_k = torch.exp(log_k)

            # Keep k between -1 and 1
            k = torch.tanh(log_k)
        elif mode == "project_degree":
            in_degree = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            in_degree = (in_degree / N) * 2 - 1
            degree = self.input_degree_project(in_degree)  # [1, N, dim]
            degree = self.input_degree_decode(degree)  # [1, N, 1]
            log_k = degree
            return log_k, degree
        elif mode == "fixed":
            k = torch.ones_like(in_adj.to_dense().sum(-1).reshape(1, -1, 1)) * 0
            k = (k / N) * 2 - 1
            return k, k
        elif mode == "project_normalized_degree":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = self.deg_mean
            var = self.deg_std
            norm_deg = (in_deg - mu) / var

            deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            deg = self.input_degree_decode(deg)  # [1, N, 1]

            # return to original domain
            unnorm_deg = (deg * var) + mu

            return unnorm_deg, unnorm_deg
        elif mode == "learn_normalized_degree":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = self.deg_mean
            var = self.deg_std
            norm_deg = (in_deg - mu) / var

            deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            deg = self.k_net(deg)  # [1, N, 1]

            # return to original domain
            unnorm_deg = (deg * var) + mu

            return unnorm_deg, unnorm_deg

        elif mode == "learn_normalized_degree_v2":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = in_deg.mean()
            var = in_deg.std()
            norm_deg = (in_deg - mu) / var

            deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            deg = self.k_net(deg)  # [1, N, 1]

            # return to original domain
            unnorm_deg = (deg * var) + mu

            return unnorm_deg, unnorm_deg

        elif mode == "learn_normalized_degree_relu":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = self.deg_mean
            var = self.deg_std
            norm_deg = (in_deg - mu) / var

            deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            deg = self.k_net(deg)  # [1, N, 1]
            # deg = F.relu(deg)   # keep it positive

            # return to original domain
            unnorm_deg = (deg * var) + mu
            unnorm_deg = F.relu(unnorm_deg)

            # add bias (so always minimum of bias)
            unnorm_deg = unnorm_deg + 1.0

            return unnorm_deg, unnorm_deg

    def perturb_edge_prob(self, log_p, noise_sample, noise):
        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_log_p = gumbel_sample(log_p, noise_sample)
        else:
            edge_log_p = log_p
        return edge_log_p

    def edge_prob_net(self, in_adj, x, mode=None):
        """

        Args:
            in_adj: [N, N]
            x: [1, N, dim]
            mode:

        Returns:

        """
        if mode is None:
            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]
            auv = in_adj.values().unsqueeze(-1).unsqueeze(0)  # [1, n, 1]
            u_v = torch.concat([u, v, auv], dim=-1)  # [1, n, dim + dim + 1]
            z = self.input_adj_project(u_v).flatten()  # [n]
            z_matrix = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z_matrix
        elif mode == "pass":
            # for debugging purposes
            z_matrix = torch.sparse.FloatTensor(
                in_adj.indices(), in_adj.values(), in_adj.shape
            )
            return z_matrix
        elif mode == "project_adj":
            auv = in_adj.values().unsqueeze(-1)  # [n, 1]
            z = self.adj_project(auv).flatten()  # [n]
            z_matrix = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z_matrix
        elif mode == "project_adj_dense":
            auv = in_adj.unsqueeze(-1)  # [N, N, 1]
            z_matrix = self.adj_project(auv).squeeze(-1).unsqueeze(0)  # [1, N, N]
            return z_matrix
        else:
            raise Exception("mode not found")


class DGG_LearnableK_debug(nn.Module):
    """
    DGG_learnableK with x_support calculated in each forward pass to account
    for varying number of input nodes (N)
    """

    def __init__(self, in_dim=32, latent_dim=64, args=None):
        super(DGG_LearnableK_debug, self).__init__()

        # torch.manual_seed(0)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.extra_edge_dim = args.extra_edge_dim
        self.extra_k_dim = args.extra_k_dim
        self.hard = args.dgg_hard
        self.deg_mean = args.deg_mean
        self.deg_std = args.deg_std

        # Edge probability network
        self.node_encode_for_edges = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self.edge_encode = nn.Sequential(
            nn.Linear(latent_dim * 2 + self.extra_edge_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.edge_conv_phi = nn.Linear(latent_dim, latent_dim // 2)
        self.edge_conv_theta = nn.Linear(latent_dim, latent_dim // 2)
        self.edge_conv_encode = nn.Linear(latent_dim // 2, 1)
        self.edge_prob_net_mode = args.dgg_mode_edge_net

        self.input_degree_decode = nn.Linear(3, 1, bias=True)
        self.combine_input_degree = nn.Sequential(
            nn.Linear(latent_dim + 3, latent_dim),
            nn.LeakyReLU(),
        )
        self.adj_project = nn.Linear(1, 1)

        # Degree estimation network
        self.k_net_mode = args.dgg_mode_k_net
        self.signal_project = nn.Linear(256, 1, bias=True)
        self.input_degree_project = nn.Linear(1, 3, bias=True)
        self.node_encode_for_k = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self.k_embed = nn.Sequential(
            nn.Linear(latent_dim + self.extra_k_dim, latent_dim // 2),
            nn.LeakyReLU(),
        )
        self.k_W = nn.Parameter(torch.rand(latent_dim, latent_dim, requires_grad=True))

        if self.k_net_mode == "input_deg":
            self.k_net = LearnableKEncoder(in_dim=3, latent_dim=latent_dim // 4)
        else:
            self.k_net = LearnableKEncoder(
                in_dim=latent_dim // 2, latent_dim=latent_dim // 4
            )

        # Top-k selector
        self.k_select_mode = args.dgg_mode_k_select

        # Gumbel noise sampler
        self.gumbel = torch.distributions.Gumbel(
            loc=torch.tensor(0.0), scale=torch.tensor(0.3)
        )

        self.var_grads = {"edge_p": [], "first_k": [], "out_adj": []}

    def hook(self, grad):
        # grad = torch.clamp(grad, min=-0.05, max=0.05)
        return grad

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

    def forward(self, x, in_adj, noise=True, writer=None, epoch=None):
        """

        Args:
            x: input points [N, dim]
            in_adj: unnormalized sparse adjacency matrix (coalesced) [N, N]
        Returns:
            adj: unnormalized sparse adjacency matrix [N, N]
        """
        assert x.ndim == 2
        assert len(in_adj.shape) == 2

        # get number of nodes
        N = x.shape[-2]

        # prepare input featuresfor rest of function
        x = x.unsqueeze(0)  # [1, N, dim]

        # print('n edges total', in_adj.to_dense().sum(), len(in_adj.indices()[0]))
        # print('in deg {:.5f} {:.5f}'.format(
        #     in_adj.to_dense().sum(-1).mean().item(), in_adj.to_dense().sum(-1).std().item()))

        # embed input and adjacency to get initial edge log probabilities
        edge_p = self.edge_prob_net(in_adj, x, mode=self.edge_prob_net_mode)  # [N, N]
        ### this module should determine the likelihood of
        ### current edged present in the graph; input features which shouldnt
        ### be connected (i.e. irrelevant edge) should have low likelihoods

        # if epoch % 1 == 0:
        #     print('e_i mu: {:.4f} std: {:.4f}'.format(
        #         edge_p.sum(-1).mean().item(), edge_p.sum(-1).std().item())
        #     )
        # perform rest of forward on dense tensors
        # print('edge p deg {:.5f} {:.5f}'.format(
        #     edge_p.sum(-1).mean().item(),
        #     edge_p.sum(-1).std().item()))

        edge_p = edge_p.unsqueeze(0)  # [1, N, N]

        # get difference between in_adj and out_adj
        # edge_p = self.get_adj_diff_stats(
        #     in_adj, edge_p, k=None, writer=writer, epoch=epoch
        # )
        # return self.return_hard_or_soft(
        #     in_adj, edge_p, idxs=None, k=None, threshold=0.5
        # )  # STEP 0

        if noise == True:
            # add gumbel noise to edge log probabilities
            edge_p = edge_p + 1e-8
            log_p = torch.log(edge_p)  # [1, N, N]
            gumbel_noise = self.gumbel.sample(log_p.shape).cuda()
            pert_log_p = gumbel_sample(log_p, gumbel_noise)
            pert_edge_p = torch.exp(pert_log_p)  # [1, N, N]
        else:
            pert_edge_p = edge_p

        pert_edge_p = self.get_adj_diff_stats(
            in_adj, pert_edge_p, k=None, writer=writer, epoch=epoch
        )
        return self.return_hard_or_soft(
            in_adj, pert_edge_p, idxs=None, k=None, threshold=0.5
        )   # STEP 1

        # get smooth top-k
        k, log_k = self.k_estimate_net(
            N, in_adj, x, pert_edge_p, mode=self.k_net_mode
        )  # [1, N, 1]
        ### maybe k should be esitmated by summing the edge probabilities for each node
        ### you dont want to gcn-deg because it passes irrelevant information to the node
        ### (from the irrelevant edges)
        ### you dont want to just use node feature x as it has no sense of whats around it
        ### getting the optimal k is about knowing which are the best neighbouring nodes

        # use pre-computed/fixed k
        # k = in_adj.to_dense().sum(-1).unsqueeze(0).unsqueeze(-1) # [1, N, 1]
        # k = torch.maximum(torch.ones_like(k), k - 1)

        # select top_k
        topk_edge_p, top_k, actual_k = self.select_top_k(
            N, k, pert_edge_p, mode=self.k_select_mode, writer=writer, epoch=epoch
        )  # [1, N, N]
        if writer is not None:
            writer.add_scalar("values/first_k_std", top_k.sum(-1).std(), epoch)
            writer.add_scalar("values/first_k_mean", top_k.sum(-1).mean(), epoch)
        # if epoch % 1 == 0:
        #     print('first_k mu: {:.4f} std: {:.4f}'.format(
        #         top_k.sum(-1).mean().item(), top_k.sum(-1).std().item())
        #     )
        debug_dict = {
            "edge_p": edge_p,  # [1, N, N]
            "first_k": top_k,  # [1, N, N]
            "out_adj": topk_edge_p,  # [1, N, N]
        }

        # print('out deg {:.5f} {:.5f}'.format(
        #     topk_edge_p.sum(-1).mean().item(),
        #     topk_edge_p.sum(-1).std().item()))

        # register hooks for gradients
        # if self.training:
        #     self.var_grads['edge_p'] = []
        #     self.var_grads['actual_k'] = []
        #     self.var_grads['out_adj'] = []
        #
        #     edge_p_grad = edge_p.register_hook(
        #         lambda grad: self.var_grads['edge_p'].append(grad)
        #     )
        #     actual_k_grad = top_k.register_hook(
        #         lambda grad: self.var_grads['actual_k'].append(grad)
        #     )
        #     topk_edge_p_grad = topk_edge_p.register_hook(
        #         lambda grad: self.var_grads['out_adj'].append(grad)
        #     )
        #     edge_p.retain_grad()
        #     actual_k.retain_grad()
        #     topk_edge_p.retain_grad()

        self.get_adj_diff_stats(
            in_adj, topk_edge_p, k, writer=writer, epoch=epoch
        )

        return self.return_hard_or_soft(in_adj, topk_edge_p, idxs=None, k=k, threshold=0.8)

    def return_hard_or_soft(self, in_adj, edge_p, idxs=None, k=None, threshold=0.8):

        # return soft
        if not self.hard:
            return edge_p.squeeze(0).to_sparse()

        # if hard adj desired, threshold first_k and scatter
        adj_hard = torch.ones_like(edge_p)

        if idxs is not None:
            adj_hard.scatter_(dim=-1, index=idxs, src=(edge_p > threshold).float())

        adj_hard = (adj_hard - edge_p).detach() + edge_p

        assert torch.any(torch.isnan(adj_hard)) == False
        assert torch.any(torch.isinf(adj_hard)) == False

        return adj_hard.to_sparse()

    def get_adj_diff_stats(
            self, in_adj, topk_edge_p=None, k=None, writer=None, epoch=None
    ):
        topk_edge_p = topk_edge_p.squeeze(0)
        in_adj = in_adj.to_dense()

        assert topk_edge_p.shape == in_adj.shape

        on_edge_mask = (in_adj > 0).float()
        off_edge_mask = (in_adj == 0).float()
        on_edge_diff = (in_adj - topk_edge_p) * on_edge_mask
        off_edge_diff = (in_adj - topk_edge_p) * off_edge_mask
        on_edge_diff_mean = on_edge_diff[on_edge_diff != 0].mean()
        on_edge_diff_std = on_edge_diff[on_edge_diff != 0].std()
        off_edge_diff_mean = off_edge_diff[off_edge_diff != 0].mean()
        off_edge_diff_std = off_edge_diff[off_edge_diff != 0].std()

        if k is not None:
            k_diff = k.flatten() - in_adj.sum(-1)
            k_diff_mean = k_diff.mean()
            k_diff_std = k_diff.std()

        if self.training:
            if writer is not None:
                writer.add_scalar("train_stats/on_edge_mean", on_edge_diff_mean, epoch)
                writer.add_scalar("train_stats/on_edge_std", on_edge_diff_std, epoch)
                writer.add_scalar("train_stats/off_edge_mean", off_edge_diff_mean, epoch)
                writer.add_scalar("train_stats/off_edge_std", off_edge_diff_std, epoch)
                writer.add_scalar("train_stats/in_deg_mean", in_adj.sum(-1).mean(),
                                  epoch)
                if k is not None:
                    writer.add_scalar("train_stats/k_diff_mean", k_diff_mean, epoch)
                    writer.add_scalar("train_stats/k_mean", k.flatten().mean(), epoch)



        return topk_edge_p

    def select_top_k(
        self, N, k, pert_edge_p, mode="k_times_edge_prob", writer=None, epoch=None
    ):
        """

        Args:
            N:
            k:
            pert_edge_p: [1, N, N]
            mode:
            writer:
            epoch:

        Returns:

        """
        if mode == "edge_p-cdf":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            # cumsum
            N = s_edge_p.shape[-1]
            s_e_cumsum = s_edge_p.cumsum(-1)  # [1, N, N]
            s_e_cumsum = s_e_cumsum / N

            # downsample to specific size
            s_e_cumsum_ds = F.interpolate(s_e_cumsum, size=[256], mode="linear")
            e_k = self.signal_project(s_e_cumsum_ds)  # [1, N, 1]
            e_k = torch.sigmoid(e_k)

            k = e_k * N

            if writer is not None:
                writer.add_scalar("values/edge_p_std", s_edge_p.sum(-1).std(), epoch)
                writer.add_scalar("values/edge_p_mean", s_edge_p.sum(-1).mean(), epoch)
                writer.add_scalar("values/k_std", k.std(), epoch)
                writer.add_scalar("values/k_mean", k.mean(), epoch)

            t = torch.arange(N).reshape(1, 1, -1).cuda()  # base domain
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1

            # Multiply sorted edge log probabilities by first-k and then softmax
            first_k_log_p = s_edge_p * first_k

            # Unsort
            adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=s_edge_p)
            return adj, first_k, k
        elif mode == "k_times_edge_prob":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            if writer is not None:
                writer.add_scalar("values/edge_p_std", s_edge_p.sum(-1).std(), epoch)
                writer.add_scalar("values/edge_p_mean", s_edge_p.sum(-1).mean(), epoch)

            t = torch.arange(N).reshape(1, 1, -1).cuda()  # base domain
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1

            # Multiply sorted edge log probabilities by first-k and then softmax
            first_k_log_p = s_edge_p * first_k

            # Unsort
            adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
            return adj, first_k, torch.tensor(0)

        elif mode == "k_only":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1

            # Unsort
            adj = first_k.clone().scatter_(dim=-1, index=idxs, src=first_k)
            return adj, first_k, torch.tensor(0)

        # k_only with linear gradients (instead of tanh saturating grads)
        elif mode == "k_only_w_linear_grad":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            first_k = -t + k

            # clamp values in forward to [0, 1] without affecting backward
            with torch.no_grad():
                first_k[:] = torch.clamp(first_k, min=0, max=1)

            # Unsort
            adj = first_k.clone().scatter_(dim=-1, index=idxs, src=first_k)
            return adj, first_k

        # k_only with linear gradients (instead of tanh saturating grads)
        elif mode == "k_times_edge_prob_w_linear_grad":
            # sort edge probabilities in DESCENDING order
            s_edge_p, idxs = torch.sort(pert_edge_p, dim=-1, descending=True)

            t = torch.arange(N, device=s_edge_p.device).reshape(1, 1, -1)  # base domain
            w = 1  # sharpness parameter
            first_k = -t + k

            # Multiply sorted edge log probabilities by first-k and then softmax
            first_k_log_p = s_edge_p * first_k

            # clamp values in forward to [0, 1] without affecting backward
            with torch.no_grad():
                first_k_log_p[:] = torch.clamp(first_k_log_p, min=0, max=1)

            # Unsort
            adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
            return adj, first_k_log_p

    def k_estimate_net(self, N, in_adj, x, edge_p, mode="calculate"):
        """

        Args:
            N: [1, N, dim]
            in_adj: unnormalized sparse adjacency matrix (coalesced) [N, N]
            x:
            edge_p: estimated edge probabilities [1, N, N]
            mode:

        Returns:

        """
        if mode == "pass":
            return None, None
        elif mode == "calculate":
            in_degree = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            k = (in_degree / N) * 2 - 1

        elif mode == "learn_normalized_degree":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = in_deg.mean()
            var = in_deg.std()
            norm_deg = (in_deg - mu) / var

            in_deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            in_deg = self.k_net(in_deg)  # [1, N, 1]

            # return to original domain
            unnorm_deg = (in_deg * var) + mu

            return unnorm_deg, unnorm_deg

        elif mode == "input_deg":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = self.deg_mean
            var = self.deg_std
            norm_deg = (in_deg - mu) / (var + 1e-5)

            in_deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            in_deg = self.k_net(in_deg)  # [1, N, 1]

            # return to original domain and relu to keep positive
            unnorm_deg = (in_deg * var) + mu
            unnorm_deg = F.relu(unnorm_deg)

            # add bias (so top k selected later is always a minimum of 1)
            unnorm_deg = unnorm_deg + 1.0

            return unnorm_deg, unnorm_deg

        elif mode == "gcn-x-deg":
            unnorm_adj_dense = in_adj.to_dense()

            # encode node features
            x = self.node_encode_for_k(x)  # [1, N, dim]

            # normalize adjacency for GCN
            norm_adj = self.normalize_adj(unnorm_adj_dense)
            norm_adj = norm_adj.unsqueeze(0)  # [1, N, N]

            # get local node feature by message-passing with neighbours
            x = norm_adj @ x @ self.k_W  # [1, N, dim]
            x = torch.relu(x)

            in_deg = unnorm_adj_dense.sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            mu = in_deg.mean()
            var = in_deg.std()
            norm_deg = (in_deg - mu) / (var + 1e-5)

            node_feats = torch.cat([x, norm_deg], dim=-1)

            # k
            deg = self.k_embed(node_feats)
            deg = self.k_net(deg)

            # return to original domain and relu to keep positive
            unnorm_deg = (deg * var) + mu
            unnorm_deg = F.relu(unnorm_deg)

            # add bias (so top k selected later is always a minimum of 1)
            unnorm_deg = unnorm_deg + 1.0

            return unnorm_deg, unnorm_deg

        elif mode == "x":
            unnorm_adj_dense = in_adj.to_dense()

            # encode node features
            x = self.node_encode_for_k(x)  # [1, N, dim]

            in_deg = unnorm_adj_dense.sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            mu = in_deg.mean()
            var = in_deg.std()
            norm_deg = (in_deg - mu) / (var + 1e-5)

            node_feats = torch.cat([x, norm_deg], dim=-1)

            # k
            deg = self.k_embed(node_feats)
            deg = self.k_net(deg)

            # return to original domain and relu to keep positive
            unnorm_deg = (deg * var) + mu
            unnorm_deg = F.relu(unnorm_deg)

            # add bias (so top k selected later is always a minimum of 1)
            unnorm_deg = unnorm_deg + 1.0

            return unnorm_deg, unnorm_deg

    def perturb_edge_prob(self, log_p, noise_sample, noise):
        if noise:
            # During training sample from Gumbel Softmax [B, N, N]
            edge_log_p = gumbel_sample(log_p, noise_sample)
        else:
            edge_log_p = log_p
        return edge_log_p

    def edge_prob_net(self, in_adj, x, mode=None):
        """

        Args:
            in_adj: sparse input adjacency [N, N]
            x: input node features [1, N, dim]
            mode:

        Returns:

        """
        if mode == "u-v-dist":
            # embed node features to lower dimension
            x = self.node_encode_for_edges(x)  # [1, N, dim]
            # print('embed x mu: {:.5f} std: {:.5f}'.format(x.mean().item(), x.std().item()))

            # get edge end features
            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]
            # print('n edges', u.shape[1])

            # distance
            t = -1.0  # this t parameter makes a significant difference
            dist = torch.linalg.vector_norm(u - v, dim=-1, ord=2)
            # print('dist edge p {:.5f} {:.5f}'.format(
            #     dist.mean().item(),
            #     dist.std().item()))
            edge_prob = torch.exp(t * dist).squeeze(0)  # [n, n]

            # convert into to sparse and then into dense
            z = torch.sparse.FloatTensor(in_adj.indices(), edge_prob, in_adj.shape)
            return z.to_dense()
        elif mode == "u-v-A_uv":
            # embed node features to lower dimension
            x = self.node_encode_for_edges(x)  # [1, N, dim]

            # get edge end features
            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]
            auv = in_adj.values().unsqueeze(-1).unsqueeze(0)  # [1, n, 1]
            edge_feat = torch.concat([u, v, auv], dim=-1)  # [1, n, dim + dim + 1]

            # edge probabilities
            z = self.edge_encode(edge_feat).flatten()  # [n]
            z = torch.sigmoid(z)

            # convert into to sparse and then into dense
            z = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z.to_dense()
        elif mode == "u-v-deg":
            # embed node features to lower dimension
            x = self.node_encode_for_edges(x)  # [1, N, dim]

            # get edge end features
            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]

            deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            mu = deg.mean()
            var = deg.std()
            norm_deg = (deg - mu) / (var + 1e-5)
            u_deg = deg[:, in_adj.indices()[0, :]]
            v_deg = deg[:, in_adj.indices()[1, :]]

            edge_feat = torch.concat(
                [u, v, u_deg, v_deg], dim=-1
            )  # [1, n, dim + dim + 2]

            # edge probabilities
            z = self.edge_encode(edge_feat).flatten()  # [n]
            z = torch.sigmoid(z)

            # convert into to sparse and then into dense
            z = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z.to_dense()
        elif mode == "u-v-deg-dist":
            # embed node features to lower dimension
            x = self.node_encode_for_edges(x)  # [1, N, dim]

            # get edge end features
            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]

            # get degree
            deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]
            mu = deg.mean()
            var = deg.std()
            norm_deg = (deg - mu) / (var + 1e-5)
            u_deg = deg[:, in_adj.indices()[0, :]]
            v_deg = deg[:, in_adj.indices()[1, :]]

            # distance
            t = -1.0  # this t parameter makes a significant difference
            dist = torch.linalg.vector_norm(u - v, dim=-1, ord=2)   # [1, N]
            edge_prob = torch.exp(t * dist).unsqueeze(-1)   # [1, N, 1]

            edge_feat = torch.concat(
                [u, v, u_deg, v_deg, edge_prob], dim=-1
            )  # [1, n, dim + dim + 3]

            # edge probabilities
            z = self.edge_encode(edge_feat).flatten()  # [n]
            z = torch.sigmoid(z)

            # convert into to sparse and then into dense
            z = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z.to_dense()
        elif mode == "edge_conv":
            # embed node features to lower dimension
            x = self.node_encode_for_edges(x)  # [1, N, dim]
            # x = F.layer_norm(x, [x.shape[-1]])

            u = x[:, in_adj.indices()[0, :]]  # [1, n, dim]
            v = x[:, in_adj.indices()[1, :]]  # [1, n, dim]
            v_u = v - u

            # edge probabilities
            edge_feat = self.edge_conv_theta(v_u) + self.edge_conv_phi(u)
            z = self.edge_conv_encode(edge_feat).flatten()
            z = torch.sigmoid(z)

            # convert into to sparse and then into dense
            z = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z.to_dense()
        elif mode == "A_uv":
            auv = in_adj.values().unsqueeze(-1)  # [n, 1]
            z = self.adj_project(auv).flatten()  # [n]
            z = torch.sigmoid(z)  # keep probs positive
            z = torch.sparse.FloatTensor(in_adj.indices(), z, in_adj.shape)
            return z.to_dense()
        else:
            raise Exception("mode not found")


class LearnableKEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, learn_k_bias=False):
        super(LearnableKEncoder, self).__init__()

        self.learn_k_bias = learn_k_bias
        # Option 1, use input to get mu, var in latent dim and
        # then project down to 1
        self.k_mu = nn.Linear(in_dim, latent_dim)
        self.k_logvar = nn.Linear(in_dim, latent_dim)
        self.k_project = nn.Linear(latent_dim, 1)

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            # return mu
            return mu

    def forward(self, x):
        latent_k_mu = self.k_mu(x)
        latent_k_logvar = self.k_logvar(x)
        latent_k = self.latent_sample(latent_k_mu, latent_k_logvar)
        k = self.k_project(latent_k)  # [B, N, 1]
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
    # x = (x / N) * 2 - 1
    y = 1 - 0.5 * (1 + torch.tanh((x - k) / w))

    plt.scatter(x, y)


def tanh_grad_test(k=0, w=1, N=10):
    print("TANH k {} w {} N {}".format(k, w, N))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = 1 - 0.5 * (1 + torch.tanh((t - k) / w))
    first_k_log_p = first_k * s_log_p
    adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
    y = adj.sum()

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    first_k_log_p.retain_grad()
    adj.retain_grad()
    k.retain_grad()

    y.backward()

    print("k", k.grad)
    print("t", t.grad)
    print("s_log_p", s_log_p.grad)
    print("first k", first_k.grad)
    print("first k log p", first_k_log_p.grad)
    print("adj", adj.grad)

    plt.scatter(
        t.detach().numpy(),
        first_k_log_p.detach().numpy(),
        label="{}-{}-{}".format(k, w, N),
    )


def tanh_grad_test_01(k=0, w=1, N=10, change=1):
    print("TANH k {} w {} N {} change {}".format(k, w, N, change))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = 1 - 0.5 * (1 + torch.tanh((t - k) / w))
    first_k_log_p = first_k * s_log_p
    adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
    gt_adj = adj.clone().detach()
    for i in range(change):
        gt_adj[i] = (gt_adj[i] - 1).abs()
    print(adj)
    print(gt_adj)

    loss = ((gt_adj - adj) ** 2).sum()
    print("loss", loss)

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    first_k_log_p.retain_grad()
    adj.retain_grad()
    k.retain_grad()

    loss.backward()

    print("k", k.grad)
    print("t", t.grad)
    print("s_log_p", s_log_p.grad)
    print("first k", first_k.grad)
    print("first k log p", first_k_log_p.grad)
    print("adj", adj.grad)
    print("\n")


def identity_grad_test(k=0, w=1, N=10):
    print("IDENTITY k {} w {} N {}".format(k, w, N))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = -t + k
    first_k_log_p = first_k * s_log_p
    with torch.no_grad():
        first_k_log_p[:] = torch.clamp(first_k_log_p, min=0, max=1)
    adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
    y = adj.sum()

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    first_k_log_p.retain_grad()
    adj.retain_grad()
    k.retain_grad()

    y.backward()

    # torch.nn.utils.clip_grad_value_(log_p, 1.0)
    # torch.nn.utils.clip_grad_value_(s_log_p, 1.0)

    print("k", k.grad)
    print("t", t.grad)
    print("s_log_p", s_log_p.grad)
    print("first k", first_k.grad)
    print("first k log p", first_k_log_p.grad)
    print("adj", adj.grad)

    plt.scatter(
        t.detach().numpy(),
        first_k_log_p.detach().numpy(),
        label="{}-{}-{}".format(k, w, N),
    )


def identity_grad_test_01(k=0, w=1, N=10, change=1):
    print("IDENTITY k {} w {} N {} change {}".format(k, w, N, change))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = -t + k
    first_k_log_p = first_k * s_log_p
    with torch.no_grad():
        first_k_log_p[:] = torch.clamp(first_k_log_p, min=0, max=1)
    adj = first_k_log_p.clone().scatter_(dim=-1, index=idxs, src=first_k_log_p)
    gt_adj = adj.clone().detach()
    for i in range(change):
        gt_adj[i] = (gt_adj[i] - 1).abs()
    print("adj", adj)
    print("gt adj", gt_adj)

    loss = ((gt_adj - adj) ** 2).sum()
    print("loss", loss)

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    first_k_log_p.retain_grad()
    adj.retain_grad()
    k.retain_grad()

    loss.backward()

    # torch.nn.utils.clip_grad_value_(log_p, 1.0)
    # torch.nn.utils.clip_grad_value_(s_log_p, 1.0)

    print("k", k)
    print("k grad", k.grad)
    print("t", t)
    print("t grad", t.grad)
    print("s_log_p", s_log_p)
    print("s_log_p grad", s_log_p.grad)
    print("first k", first_k)
    print("first k grad", first_k.grad)
    print("first k log p", first_k_log_p)
    print("first k log p grad", first_k_log_p.grad)
    print("adj", adj)
    print("adj grad", adj.grad)
    print("\n")

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1])
    ax50 = fig.add_subplot(gs[5, 0])
    ax51 = fig.add_subplot(gs[5, 1])
    ax60 = fig.add_subplot(gs[6, 0])
    ax61 = fig.add_subplot(gs[6, 1])
    ax70 = fig.add_subplot(gs[7, 0])
    ax71 = fig.add_subplot(gs[7, 1])

    ax00.imshow(adj.detach().numpy().reshape(1, -1))
    ax10.imshow(gt_adj.detach().numpy().reshape(1, -1))
    ax20.imshow(k.detach().numpy().reshape(1, -1))
    ax21.imshow(k.grad.detach().numpy().reshape(1, -1))
    ax30.imshow(t.detach().numpy().reshape(1, -1))
    ax31.imshow(t.grad.detach().numpy().reshape(1, -1))
    ax40.imshow(s_log_p.detach().numpy().reshape(1, -1))
    ax41.imshow(s_log_p.grad.detach().numpy().reshape(1, -1))
    ax50.imshow(first_k.detach().numpy().reshape(1, -1))
    ax51.imshow(first_k.grad.detach().numpy().reshape(1, -1))
    ax60.imshow(first_k_log_p.detach().numpy().reshape(1, -1))
    ax61.imshow(first_k_log_p.grad.detach().numpy().reshape(1, -1))
    ax70.imshow(adj.detach().numpy().reshape(1, -1))
    ax71.imshow(adj.grad.detach().numpy().reshape(1, -1))

    ax00.set_title("adj")
    ax10.set_title("gt_adj")
    ax20.set_title("k")
    ax30.set_title("t")
    ax40.set_title("s_log_p")
    ax50.set_title("first_k")
    ax60.set_title("first_k_log_p")
    ax70.set_title("adj")

    plt.show()


def identity_grad_test_02(k=0, w=1, N=10, change=1):
    print("IDENTITY k {} w {} N {} change {}".format(k, w, N, change))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = -t + k
    with torch.no_grad():
        first_k[:] = torch.clamp(first_k, min=0, max=1)
    adj = first_k.clone().scatter_(dim=-1, index=idxs, src=first_k)
    gt_adj = adj.clone().detach()
    for i in range(change):
        gt_adj[i] = (gt_adj[i] - 1).abs()
    print("adj", adj)
    print("gt adj", gt_adj)

    loss = ((gt_adj - adj) ** 2).sum()
    print("loss", loss)

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    adj.retain_grad()
    k.retain_grad()

    loss.backward()

    # torch.nn.utils.clip_grad_value_(log_p, 1.0)
    # torch.nn.utils.clip_grad_value_(s_log_p, 1.0)

    print("k", k)
    print("k grad", k.grad)
    print("t", t)
    print("t grad", t.grad)
    print("s_log_p", s_log_p)
    print("first k", first_k)
    print("first k grad", first_k.grad)
    print("adj", adj)
    print("adj grad", adj.grad)
    print("\n")

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1])
    ax50 = fig.add_subplot(gs[5, 0])
    ax51 = fig.add_subplot(gs[5, 1])
    ax60 = fig.add_subplot(gs[6, 0])
    ax61 = fig.add_subplot(gs[6, 1])
    ax70 = fig.add_subplot(gs[7, 0])
    ax71 = fig.add_subplot(gs[7, 1])

    ax00.imshow(adj.detach().numpy().reshape(1, -1))
    ax10.imshow(gt_adj.detach().numpy().reshape(1, -1))
    ax20.imshow(k.detach().numpy().reshape(1, -1))
    ax21.imshow(k.grad.detach().numpy().reshape(1, -1))
    ax30.imshow(t.detach().numpy().reshape(1, -1))
    ax31.imshow(t.grad.detach().numpy().reshape(1, -1))
    ax40.imshow(s_log_p.detach().numpy().reshape(1, -1))
    ax50.imshow(first_k.detach().numpy().reshape(1, -1))
    ax51.imshow(first_k.grad.detach().numpy().reshape(1, -1))
    ax70.imshow(adj.detach().numpy().reshape(1, -1))
    ax71.imshow(adj.grad.detach().numpy().reshape(1, -1))

    ax00.set_title("adj")
    ax10.set_title("gt_adj")
    ax20.set_title("k")
    ax30.set_title("t")
    ax40.set_title("s_log_p")
    ax50.set_title("first_k")
    ax60.set_title("first_k_log_p")
    ax70.set_title("adj")

    plt.show()


def identity_grad_test_03(k=3, w=1, N=10, gt_k=8):
    print("IDENTITY k {} w {} N {} gt_k {}".format(k, w, N, gt_k))
    torch.manual_seed(0)
    k = torch.tensor(k, requires_grad=True)
    t = torch.arange(N, requires_grad=True)
    log_p = (torch.rand(int(N)) > 0.5).float()
    log_p.requires_grad_(True)
    s_log_p, idxs = torch.sort(log_p, descending=True)
    first_k = -t + k
    with torch.no_grad():
        first_k[:] = torch.clamp(first_k, min=0, max=1)

    gt_first_k = torch.zeros_like(first_k)
    gt_first_k[: int(gt_k)] = 1
    loss = (gt_first_k - first_k) ** 2
    loss = loss.sum()
    print("loss", loss)

    t.retain_grad()
    log_p.retain_grad()
    s_log_p.retain_grad()
    first_k.retain_grad()
    k.retain_grad()

    loss.backward()

    # torch.nn.utils.clip_grad_value_(log_p, 1.0)
    # torch.nn.utils.clip_grad_value_(s_log_p, 1.0)

    print("k", k)
    print("k grad", k.grad)
    print("t", t)
    print("t grad", t.grad)
    print("s_log_p", s_log_p)
    print("first k", first_k)
    print("first k grad", first_k.grad)
    print("\n")

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1])
    ax50 = fig.add_subplot(gs[5, 0])
    ax51 = fig.add_subplot(gs[5, 1])
    ax60 = fig.add_subplot(gs[6, 0])
    ax61 = fig.add_subplot(gs[6, 1])
    ax70 = fig.add_subplot(gs[7, 0])
    ax71 = fig.add_subplot(gs[7, 1])

    ax20.imshow(k.detach().numpy().reshape(1, -1))
    ax21.imshow(k.grad.detach().numpy().reshape(1, -1))
    ax30.imshow(t.detach().numpy().reshape(1, -1))
    ax31.imshow(t.grad.detach().numpy().reshape(1, -1))
    ax40.imshow(s_log_p.detach().numpy().reshape(1, -1))
    ax50.imshow(first_k.detach().numpy().reshape(1, -1))
    ax51.imshow(first_k.grad.detach().numpy().reshape(1, -1))

    ax00.set_title("adj")
    ax10.set_title("gt_adj")
    ax20.set_title("k")
    ax30.set_title("t")
    ax40.set_title("s_log_p")
    ax50.set_title("first_k")
    ax60.set_title("first_k_log_p")
    ax70.set_title("adj")

    plt.show()


if __name__ == "__main__":

    # tanh_test_exp(2.5)
    # tanh_test(12.2)

    # tanh_plot(k=0, w=1)
    # tanh_grad_test(k=1.0, w=1, N=7.0)
    # identity_grad_test(k=1.0, w=1, N=7.0)
    # plt.legend()
    # plt.show()
    #
    # tanh_grad_test(k=3.0, w=1, N=7.0)
    # identity_grad_test(k=3.0, w=1, N=7.0)
    # plt.legend()
    # plt.show()
    #
    # tanh_grad_test(k=7.0, w=1, N=7.0)
    # identity_grad_test(k=7.0, w=1, N=7.0)
    # plt.legend()
    # plt.show()

    # identity_grad_test_01(k=1.0, w=1, N=7.0, change=0)
    # tanh_grad_test_01(k=1.0, w=1, N=7.0, change=0)
    # identity_grad_test_01(k=1.0, w=1, N=7.0, change=1)
    # tanh_grad_test_01(k=1.0, w=1, N=7.0, change=1)
    # identity_grad_test_01(k=1.0, w=1, N=7.0, change=3)
    # tanh_grad_test_01(k=1.0, w=1, N=7.0, change=3)
    # identity_grad_test_01(k=1.0, w=1, N=7.0, change=5)

    # identity_grad_test_01(k=1.0, w=1, N=7.0, change=5)
    # identity_grad_test_02(k=1.0, w=1, N=7.0, change=5)
    #
    # identity_grad_test_01(k=6.0, w=1, N=7.0, change=5)
    # identity_grad_test_02(k=6.0, w=1, N=7.0, change=5)

    identity_grad_test_03(k=2.0, w=1, N=10.0, gt_k=8.0)
    identity_grad_test_03(k=8.0, w=1, N=10.0, gt_k=2.0)
