"""View-selection schedule and acquisition functions (official active/schema.py, H_reg.py,
rand_selector.py of github.com/JiangWenPL/FisherRF @ b747328)."""

import random

import torch


def schedule(n_total, per_round, num_init, interval_epochs):
    """VNSeqMInplace.load_its: {iteration: number of views to add}."""
    if n_total < num_init or (n_total - num_init) % per_round:
        raise ValueError("cannot split the remaining views evenly into rounds")
    load, cur, left = {}, num_init, n_total - num_init
    it_base = cur * interval_epochs
    while left > 0:
        load[it_base] = per_round
        cur += per_round
        it_base += cur * interval_epochs
        left -= per_round
    return load


def initial_views(centers, num_init):
    """VNSeqMInplace init: training view 0, then greedy farthest camera centre (max-min)."""
    centers = centers.detach().float().cpu()
    init = [0]
    candidates = [i for i in range(len(centers)) if i not in init]
    while len(init) < num_init:
        dist = torch.cdist(centers[candidates], centers[init]).min(dim=1).values
        init.append(candidates.pop(int(dist.argmax())))
    return init


def eig_scores(h_train, h_candidates, reg_lambda):
    """Eq. 5/8: tr(H_cand (diag H_train + lambda I)^-1) = sum_k H_cand[k] / (H_train[k] + lambda)."""
    inv = torch.reciprocal(h_train + reg_lambda)
    return torch.stack([(h * inv).sum() for h in h_candidates])


def greedy_batch(h_train, h_candidates, candidates, num_views, reg_lambda):
    """Alg. 1: greedy batch selection, H_b = H_{b-1} + H_i after each pick (H_reg.nbvs)."""
    h_train = h_train.clone()
    h_candidates, candidates = list(h_candidates), list(candidates)
    picked = []
    for _ in range(num_views):
        scores = eig_scores(h_train, h_candidates, reg_lambda)
        i = int(scores.argmax())
        picked.append(candidates.pop(i))
        h_train += h_candidates.pop(i)
    return picked


def random_select(candidates, num_views, seed=0):
    """RandSelector.nbvs: shuffle the sorted candidate list with random.Random(seed)."""
    candidates = list(candidates)
    random.Random(seed).shuffle(candidates)
    return candidates[:num_views]
