"""C3 significance-weighted vector quantisation of SH, ported from the official code.

Official sources (Inria Gaussian-Splatting license): `vectree/vq.py` (EuclideanCodebook,
kmeans_init=False, decay 0.8, eps 1e-5, threshold_ema_dead_code 0, commitment unused) and
`vectree/vectree.py` (Quantization.quantize / calc_vector_quantized_feature).
The one-hot einsums of the official forward are written as index_add (same sums, less memory).
"""

import math

import torch


def kaiming_bound(codebook_size, dim):
    """nn.init.kaiming_uniform_ on a (1, K, D) tensor: fan_in = K * D, gain sqrt(2)."""
    return math.sqrt(2.0) * math.sqrt(3.0 / (codebook_size * dim))


def init_codebook(codebook_size, dim, generator, device):
    b = kaiming_bound(codebook_size, dim)
    u = torch.rand(codebook_size, dim, generator=generator, device=device)
    return (u * 2 - 1) * b


@torch.no_grad()
def nearest(x, embed, chunk=8192):
    """argmax(-cdist(x, embed)) (official gumbel_sample with temperature 0), chunked."""
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = torch.cdist(x[s:s + chunk], embed, p=2).argmin(dim=1)
    return out


@torch.no_grad()
def ema_step(embed, cluster_size, x, weight, decay=0.8, eps=1e-5):
    """One training-mode EuclideanCodebook.forward with sample weights (in place).

    weight <- weight * n / sum(weight); cs <- decay cs + (1-decay) sum_onehot w;
    embed <- decay embed + (1-decay) (sum_onehot w x) / (laplace(cs) * sum(cs)).
    Returns the assignment of `x` (computed with the codebook before the update).
    """
    k = embed.shape[0]
    w = weight * weight.numel() / weight.sum()
    ind = nearest(x, embed)
    batch_cs = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, ind, w)
    cluster_size.mul_(decay).add_(batch_cs, alpha=1 - decay)
    embed_sum = torch.zeros_like(embed).index_add_(0, ind, x * w[:, None])
    smoothed = (cluster_size + eps) / (cluster_size.sum() + k * eps) * cluster_size.sum()
    embed.mul_(decay).add_(embed_sum / smoothed[:, None], alpha=1 - decay)
    return ind


@torch.no_grad()
def learn_codebook(feats, importance, codebook_size=8192, iterations=1000, batch=80000,
                   decay=0.8, k_expire=10, generator=None):
    """`Quantization.quantize` codebook loop on the features selected for VQ.

    Each iteration samples `batch` rows with replacement, applies one weighted EMA step and then
    overwrites the `k_expire` codes with the smallest EMA cluster size by the `k_expire` most
    important sampled vectors.
    """
    dev = feats.device
    feats = feats.float()
    imp = importance.float()
    embed = init_codebook(codebook_size, feats.shape[1], generator, dev)
    cluster_size = torch.zeros(codebook_size, device=dev)
    k = k_expire if k_expire <= codebook_size else 0
    for _ in range(int(iterations)):
        idx = torch.randint(0, len(feats), (batch,), generator=generator, device=dev)
        w, x = imp[idx], feats[idx]
        ema_step(embed, cluster_size, x, w, decay)
        if k > 0:
            replace_index = torch.topk(cluster_size, k=k, largest=False).indices
            most_important = torch.topk(w, k=k, largest=True).indices
            embed[replace_index] = x[most_important]
    return embed, cluster_size


@torch.no_grad()
def split_masks(importance, vq_ratio=0.6):
    """Non-VQ = top int(N (1 - vq_ratio)) by importance (`torch.topk(largest=True)`)."""
    n = len(importance)
    keep = torch.topk(importance, k=int(n * (1 - vq_ratio)), largest=True).indices
    non_vq = torch.zeros(n, dtype=torch.bool, device=importance.device)
    non_vq[keep] = True
    return non_vq


# --- feature layout (official PLY order) ---------------------------------------------------------

def sh_features(sh0, shN, rest_coeffs=8):
    """[N, 3 + 3 * rest_coeffs] = f_dc_0..2, f_rest channel-major (PLY order of the official
    `construct_list_of_attributes` for SH degree 2)."""
    n = len(sh0)
    dc = sh0.reshape(n, 3)
    rest = shN[:, :rest_coeffs, :].transpose(1, 2).reshape(n, 3 * rest_coeffs)
    return torch.cat((dc, rest), dim=1)


def from_sh_features(feats, rest_coeffs=8):
    """Inverse of `sh_features`: (sh0 [N,1,3], shN_rest [N,rest_coeffs,3])."""
    n = len(feats)
    sh0 = feats[:, :3].reshape(n, 1, 3)
    rest = feats[:, 3:].reshape(n, 3, rest_coeffs).transpose(1, 2)
    return sh0, rest
