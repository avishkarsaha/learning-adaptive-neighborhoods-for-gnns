import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.t = nn.Parameter(torch.tensor(-0.1))
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

        if (
            self.k_net_mode == "input_deg"
            or self.k_net_mode == "learn_normalized_degree"
        ):
            self.k_net = LearnableKEncoder(
                in_dim=3, latent_dim=latent_dim // 4, args=args
            )
        else:
            self.k_net = LearnableKEncoder(
                in_dim=latent_dim // 2, latent_dim=latent_dim // 4, args=args
            )

        # Top-k selector
        self.k_select_mode = args.dgg_mode_k_select

        # Gumbel noise sampler
        self.gumbel = torch.distributions.Gumbel(
            loc=torch.tensor(0.0), scale=torch.tensor(0.3)
        )

        self.var_grads = {"edge_p": [], "first_k": [], "out_adj": []}

        self.args = args

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

        # embed input and adjacency to get initial edge log probabilities
        edge_p = self.edge_prob_net(in_adj, x, mode=self.edge_prob_net_mode)  # [N, N]

        # perform rest of forward on dense tensors
        edge_p = edge_p.unsqueeze(0)  # [1, N, N]

        if self.args.debug_step == 0:
            # get difference between in_adj and out_adj
            edge_p = self.get_adj_diff_stats(
                in_adj, edge_p, k=None, writer=writer, epoch=epoch
            )
            return self.return_hard_or_soft(
                in_adj, edge_p, idxs=None, k=None, threshold=0.5
            )  # STEP 0

        if self.args.perturb_edge_prob:
            # add gumbel noise to edge log probabilities
            edge_p = edge_p + 1e-8
            log_p = torch.log(edge_p)  # [1, N, N]

            if self.args.symmetric_noise:
                # add a symmetric gumbel noise matrix
                G = torch.zeros_like(edge_p).squeeze(0)  # [N, N]
                i, j = torch.triu_indices(G.shape[0], G.shape[1], 1)
                gumbel_noise = self.gumbel.sample([len(i)]).cuda()
                G[i, j] = gumbel_noise
                G.T[i, j] = gumbel_noise
                G = G.unsqueeze(0)  # [1, N, N]
            else:
                # asymmetric gumbel noise matrix
                G = self.gumbel.sample(log_p.shape).cuda()

            pert_log_p = gumbel_sample(log_p, G)
            pert_edge_p = torch.exp(pert_log_p)  # [1, N, N]
        else:
            pert_edge_p = edge_p

        if self.args.debug_step == 1:
            pert_edge_p = self.get_adj_diff_stats(
                in_adj, pert_edge_p, k=None, writer=writer, epoch=epoch
            )
            return self.return_hard_or_soft(
                in_adj, pert_edge_p, idxs=None, k=None, threshold=0.5
            )  # STEP 1

        # get smooth top-k
        k, _ = self.k_estimate_net(
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
        topk_edge_p, top_k, idxs = self.select_top_k(
            N, k, pert_edge_p, mode=self.k_select_mode, writer=writer, epoch=epoch
        )  # [1, N, N]
        if writer is not None:
            writer.add_scalar("values/first_k_std", top_k.sum(-1).std(), epoch)
            writer.add_scalar("values/first_k_mean", top_k.sum(-1).mean(), epoch)

        debug_dict = {
            "edge_p": edge_p,  # [1, N, N]
            "first_k": top_k,  # [1, N, N]
            "out_adj": topk_edge_p,  # [1, N, N]
        }

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

        self.get_adj_diff_stats(in_adj, topk_edge_p, k, writer=writer, epoch=epoch)

        return self.return_hard_or_soft(
            in_adj, topk_edge_p, idxs=idxs, k=k, threshold=0.5
        )

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

        return adj_hard.squeeze(0).to_sparse()

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
                writer.add_scalar(
                    "train_stats/off_edge_mean", off_edge_diff_mean, epoch
                )
                writer.add_scalar("train_stats/off_edge_std", off_edge_diff_std, epoch)
                writer.add_scalar(
                    "train_stats/in_deg_mean", in_adj.sum(-1).mean(), epoch
                )
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
            return adj, first_k, idxs

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
            return k

        elif mode == "learn_normalized_degree":
            in_deg = in_adj.to_dense().sum(-1).reshape(1, -1, 1)  # [1, N, 1]

            mu = in_deg.mean()
            var = in_deg.std()
            norm_deg = (in_deg - mu) / var

            in_deg = self.input_degree_project(norm_deg)  # [1, N, dim]
            in_deg = self.k_net(in_deg)  # [1, N, 1]

            # return to original domain
            unnorm_deg = (in_deg * var) + mu
            unnorm_deg = F.relu(unnorm_deg)
            unnorm_deg = unnorm_deg + 1.0

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
            t = -0.05  # this t parameter makes a significant difference
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
            dist = torch.linalg.vector_norm(u - v, dim=-1, ord=2)  # [1, N]
            edge_prob = torch.exp(t * dist).unsqueeze(-1)  # [1, N, 1]

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


class DGG(nn.Module):
    """
    Differentiable graph generator for ICLR
    """

    def __init__(self, in_dim=32, latent_dim=64, args=None):
        super().__init__()

        self.args = args

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )

        # Edge ranker
        self.edge_encoder = nn.Sequential(
            nn.Linear(latent_dim + self.args.extra_edge_dim, latent_dim), nn.LeakyReLU()
        )

        # Degree estimator
        self.degree_decoder = nn.Sequential(nn.Linear(1, 1, bias=True), nn.LeakyReLU())

        # Top-k selector

        self.var_grads = {"edge_p": [], "first_k": [], "out_adj": []}

    def forward(self, x, adj, noise=True, writer=None, epoch=None):
        """

        Args:
            x:
            adj:
            noise:
            writer:
            epoch:

        Returns:

        """

        assert x.ndim == 2
        assert len(adj.shape) == 2

        N = x.shape[0]  # number of nodes

        # Encode node features [N, in_dim] ---> [N, h]
        x = self.node_encoder(x)

        # Rank edges using encoded node features [N, h] ----> [E]
        u = x[adj.indices()[0, :]]  # [E, dim]
        v = x[adj.indices()[1, :]]  # [E, dim]
        uv_diff = u - v
        edge_feat = self.edge_encoder(uv_diff)
        edge_rank = edge_feat.sum(-1)  # [E]
        edge_rank = torch.sigmoid(edge_rank)
        edge_rank = torch.sparse.FloatTensor(adj.indices(), edge_rank, adj.shape)
        edge_rank = edge_rank.to_dense()

        # Estimate node degree using encoded node features and edge rankings
        k = edge_rank.sum(-1).unsqueeze(-1)
        k = self.degree_decoder(k)  # [N, 1]

        # Select top-k edges
        # sort edge ranks descending
        srt_edge_rank, idxs = torch.sort(edge_rank, dim=-1, descending=True)

        t = torch.arange(N).reshape(1, N).cuda()  # base domain [1, N]
        # k = k.unsqueeze(0)                          # [N, 1]
        w = 1  # sharpness parameter
        first_k = 1 - 0.5 * (
            1 + torch.tanh((t - k) / w)
        )  # higher k = more items closer to 1
        first_k = first_k + 1.0

        # Multiply edge rank by first k
        first_k_ranks = srt_edge_rank * first_k

        # Unsort
        out_adj = first_k_ranks.clone().scatter_(dim=-1, index=idxs, src=first_k_ranks)
        # out_adj = out_adj[adj.indices()[0, :], adj.indices()[1, :]]
        # out_adj = torch.sparse.FloatTensor(adj.indices(), out_adj, adj.shape)

        # return top-k edges and encoded node features
        return out_adj.to_sparse(), x

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

        return adj_hard.squeeze(0).to_sparse()

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
                writer.add_scalar(
                    "train_stats/off_edge_mean", off_edge_diff_mean, epoch
                )
                writer.add_scalar("train_stats/off_edge_std", off_edge_diff_std, epoch)
                writer.add_scalar(
                    "train_stats/in_deg_mean", in_adj.sum(-1).mean(), epoch
                )
                if k is not None:
                    writer.add_scalar("train_stats/k_diff_mean", k_diff_mean, epoch)
                    writer.add_scalar("train_stats/k_mean", k.flatten().mean(), epoch)

        return topk_edge_p


class DGG_Ablations(nn.Module):
    """
    Differentiable graph generator for ICLR
    """

    def __init__(self, in_dim=32, latent_dim=64, args=None):
        super().__init__()

        self.args = args

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LeakyReLU(),
        )

        # Edge ranker
        self.edge_encoder = nn.Sequential(
            nn.Linear(latent_dim + self.args.extra_edge_dim, latent_dim), nn.LeakyReLU()
        )

        # Degree estimator
        self.degree_decoder = nn.Sequential(nn.Linear(1, 1, bias=True), nn.LeakyReLU())

        # Top-k selector

        self.var_grads = {"edge_p": [], "first_k": [], "out_adj": []}

    def forward(self, x, adj, k=None, writer=None, epoch=None):
        """

        Args:
            x:
            adj:
            noise:
            writer:
            epoch:

        Returns:

        """

        assert x.ndim == 2
        assert len(adj.shape) == 2

        N = x.shape[0]  # number of nodes

        # Encode node features [N, in_dim] ---> [N, h]
        x = self.node_encoder(x)

        # Rank edges using encoded node features [N, h] ----> [E]
        u = x[adj.indices()[0, :]]  # [E, dim]
        v = x[adj.indices()[1, :]]  # [E, dim]
        uv_diff = u - v
        edge_feat = self.edge_encoder(uv_diff)
        edge_rank = edge_feat.sum(-1)  # [E]
        edge_rank = torch.sigmoid(edge_rank)
        noise = torch.rand(edge_rank.shape, device=x.device) * 2 - 1
        edge_rank = edge_rank + noise
        edge_rank = torch.sigmoid(edge_rank)
        edge_rank = torch.sparse.FloatTensor(adj.indices(), edge_rank, adj.shape)
        edge_rank = edge_rank.to_dense()

        # Select top-k edges
        # sort edge ranks descending
        srt_edge_rank, idxs = torch.sort(edge_rank, dim=-1, descending=True)

        if k is not None:
            srt_edge_rank[:, k:] = 0
            first_k_ranks = srt_edge_rank
        else:
            # Estimate node degree using encoded node features and edge rankings
            k = edge_rank.sum(-1).unsqueeze(-1)
            k = self.degree_decoder(k)  # [N, 1]

            t = torch.arange(N).reshape(1, N).cuda()  # base domain [1, N]
            # k = k.unsqueeze(0)                          # [N, 1]
            w = 1  # sharpness parameter
            first_k = 1 - 0.5 * (
                1 + torch.tanh((t - k) / w)
            )  # higher k = more items closer to 1
            first_k = first_k + 1.0

            # Multiply edge rank by first k
            first_k_ranks = srt_edge_rank * first_k

        # Unsort
        out_adj = first_k_ranks.clone().scatter_(dim=-1, index=idxs, src=first_k_ranks)
        # out_adj = out_adj[adj.indices()[0, :], adj.indices()[1, :]]
        # out_adj = torch.sparse.FloatTensor(adj.indices(), out_adj, adj.shape)

        # return top-k edges and encoded node features
        return out_adj.to_sparse(), x

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

        return adj_hard.squeeze(0).to_sparse()

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
                writer.add_scalar(
                    "train_stats/off_edge_mean", off_edge_diff_mean, epoch
                )
                writer.add_scalar("train_stats/off_edge_std", off_edge_diff_std, epoch)
                writer.add_scalar(
                    "train_stats/in_deg_mean", in_adj.sum(-1).mean(), epoch
                )
                if k is not None:
                    writer.add_scalar("train_stats/k_diff_mean", k_diff_mean, epoch)
                    writer.add_scalar("train_stats/k_mean", k.flatten().mean(), epoch)

        return topk_edge_p


class LearnableKEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, learn_k_bias=False, args=None):
        super(LearnableKEncoder, self).__init__()

        self.learn_k_bias = learn_k_bias
        # Option 1, use input to get mu, var in latent dim and
        # then project down to 1
        self.k_mu = nn.Linear(in_dim, latent_dim)
        self.k_logvar = nn.Linear(in_dim, latent_dim)
        self.k_project = nn.Linear(latent_dim, 1)
        self.args = args

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

        if self.args.stochastic_k:
            # stochastic k estimation with a latent sample
            latent_k_mu = self.k_mu(x)
            latent_k_logvar = self.k_logvar(x)
            latent_k = self.latent_sample(latent_k_mu, latent_k_logvar)
        else:
            # deterministic k estimation
            latent_k = self.k_mu(x)

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