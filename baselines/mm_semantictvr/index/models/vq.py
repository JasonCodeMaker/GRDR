import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.55, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100, use_linear=0,
                 ema_update=True, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.ema_update = ema_update
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

        # EMA buffers for codebook statistics
        if ema_update:
            self.register_buffer('cluster_size', torch.ones(n_e))
            self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def laplace_smoothing(self, x, n_categories, eps=1e-5):
        """Smooth cluster counts to prevent division by zero"""
        return (x + eps) / (x.sum() + n_categories * eps) * x.sum()

    @torch.no_grad()
    def update_ema(self):
        """Update embedding from accumulated EMA statistics"""
        cluster_size = self.laplace_smoothing(self.cluster_size, self.n_e, self.eps)
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(-1)
        self.embedding.weight.data.copy_(embed_normalized)

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)
            
        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, embeddings_weight.t())
        use_sinkhorn = use_sk and self.sk_epsilon > 0 and not self.ema_update

        if not use_sinkhorn:
            indices = torch.argmin(d, dim=-1)
            # print("=======",self.sk_epsilon)
        else:
            # print("++++++++",self.sk_epsilon)
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            # print(Q.sum(0)[:10])
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)
        if self.use_linear == 1:
            x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        else:
            x_q = self.embedding(indices).view(x.shape)

        # EMA statistics accumulation (training only)
        if self.training and self.ema_update:
            with torch.no_grad():
                indices_flat = indices.view(-1)
                encodings = F.one_hot(indices_flat, self.n_e).float()

                # Update cluster size (usage count)
                cluster_size = encodings.sum(0)
                self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                # Update embedding average
                embed_sum = latent.t() @ encodings
                self.embed_avg.data.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

            # Apply EMA update to codebook
            self.update_ema()

        # compute loss for embedding
        if self.ema_update:
            # EMA mode: commitment loss only (no codebook gradients)
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = self.beta * commitment_loss
        else:
            # Gradient mode: original loss
            commitment_loss = F.mse_loss(x_q.detach(), x)
            codebook_loss = F.mse_loss(x_q, x.detach())
            loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d


