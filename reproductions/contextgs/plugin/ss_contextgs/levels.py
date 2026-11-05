"""Anchor partitioning (Eq. 4-6) and the anchor-level context model (Eq. 7-8, 11).

Line-by-line ports of the official functions (github.com/wyf0912/ContextGS @ eff8d19):
utils/multi_level.py:torch_unique_with_indices, scene/gaussian_model.py:find_divide_scale, divide_levels,
mapping_to_orign, index_of_level_L_in_orign, extract_context_feat and multi_scale_generating. The index
bookkeeping is kept verbatim, including two official behaviours that differ from the paper text (paper.md §7):
the context carries the *parent's* position, and the level-1 context rows are gathered in increasing
original-index order while the level-1 anchors are enumerated in voxel order.
"""

import torch

from . import model


def torch_unique_with_indices(tensor):
    """Unique rows, inverse indices, the minimum original index per unique row (M in Eq. 5), counts."""
    with torch.no_grad():
        unique, inverse, counts = torch.unique(tensor, return_inverse=True, dim=0,
                                               return_counts=True)
        indices = torch.scatter_reduce(
            input=torch.zeros_like(unique[:, 0], dtype=torch.double),
            src=torch.arange(tensor.size(0), device=tensor.device, dtype=torch.double),
            dim=0, index=inverse, reduce="amin", include_self=False).long()
        return unique, inverse, indices, counts


@torch.no_grad()
def find_divide_scale(anchor, voxel_size, x_bound_min, x_bound_max, target_ratio, level_num):
    """Binary search of kappa_k so that |V^{k+1}| / |V^k| ~ tau (§4.1)."""
    scale_upper = ((x_bound_max - x_bound_min) / voxel_size).max()

    def binary_search(scale_upper, scale_lower, anchor):
        while True:
            scale = (scale_upper + scale_lower) / 2
            anchor_unique = torch.unique(torch.round(anchor / voxel_size / scale),
                                         dim=0) * voxel_size * scale
            ratio = anchor_unique.shape[0] / anchor.shape[0]
            if abs(ratio - target_ratio) < 0.01 or abs(scale_upper - scale_lower) < 1:
                break
            if ratio < target_ratio:
                scale_upper = scale
            else:
                scale_lower = scale
        return scale, anchor_unique

    anchor_unique = anchor
    scale_list = []
    scale_lower = 1
    for _ in range(level_num - 1):
        scale, anchor_unique = binary_search(scale_upper, scale_lower, anchor_unique)
        scale_lower = scale
        scale_list.append(float(scale))
    return scale_list


def divide_levels(anchor, voxel_size, level_scale, level_num, mask_anchor_bool=None):
    """Eq. 4-5: V-hat^k from V-hat^{k-1}, representative = minimum index of each level-k voxel."""
    hybrid_anchor_list = [anchor]
    inverse_indices_list, mapping_list = [], []
    hybrid_anchor = anchor
    for i in range(1, level_num):
        if i == 1 and mask_anchor_bool is not None:
            hybrid_anchor = hybrid_anchor * mask_anchor_bool.unsqueeze(1)
        _, inverse, mapping, _ = torch_unique_with_indices(
            torch.round(hybrid_anchor / voxel_size / level_scale[i - 1]))
        hybrid_anchor = hybrid_anchor[mapping]
        hybrid_anchor_list.append(hybrid_anchor)
        inverse_indices_list.append(inverse)
        mapping_list.append(mapping)
    return hybrid_anchor_list, inverse_indices_list, mapping_list, hybrid_anchor


def mapping_to_orign(mapping_list, L, mask=None):
    """Index of level-L entries (optionally a subset) in the original anchor space."""
    assert L > 0
    level = L - 1
    prev = mapping_list[level] if mask is None else mapping_list[level][mask]
    for i in reversed(range(level)):
        prev = mapping_list[i][prev]
    return prev


def index_of_level_L_in_orign(mapping_list, inverse_indices_list, idx, L):
    tmp = idx
    for i in range(L):
        tmp = inverse_indices_list[i][tmp]
    return mapping_to_orign(mapping_list, L, mask=tmp)


def extract_context_feat(anchor, feat_q, scaling_q, already_coded, inverse_indices_list,
                         mapping_list, i):
    """[x, f, l] of the already coded level-i parent of every to-be-coded anchor of level i-1 (Eq. 8)."""
    content = torch.cat([anchor, feat_q, scaling_q], dim=1)
    if i > 1:
        gather = torch.zeros_like(content[:, 0]).bool()
        gather[mapping_to_orign(mapping_list, i - 1)] = True
    else:
        gather = torch.ones_like(content[:, 0]).bool()
    gather = gather & (~already_coded)
    last = index_of_level_L_in_orign(mapping_list, inverse_indices_list,
                                     torch.nonzero(gather)[:, 0], i)
    return torch.gather(content, 0, last[:, None].expand(-1, content.shape[1]))


def level_inputs(levels, i, content_pre, hyper_feat, n_anchor):
    """to-code mask of level i, its original indices, and the MLP input [x or context, z-hat]."""
    hybrid_anchor_list, _, mapping_list, _ = levels
    level_num = len(hybrid_anchor_list)
    if i != level_num - 1:
        to_code = torch.ones_like(hybrid_anchor_list[i][:, 0]).bool()
        to_code[mapping_list[i]] = False
    else:
        to_code = torch.ones_like(hybrid_anchor_list[i][:, 0]).bool()
    if i != 0:
        orig = mapping_to_orign(mapping_list, i, to_code)
    else:
        orig = torch.arange(n_anchor, device=to_code.device)[to_code]
    if content_pre is None:
        feat_in = torch.cat([hybrid_anchor_list[i][to_code], hyper_feat[orig].float()], dim=1)
    else:
        feat_in = torch.cat([content_pre, hyper_feat[orig]], dim=1)
    return to_code, orig, feat_in


def multi_scale(bank, anchor, hyper, feat, offsets, scaling, masks, mask_anchor_bool=None,
                training=False, predict_bpp=False, x_means=None, sample=0.15, use_hyper=True):
    """multi_scale_generating: code levels K-1 .. 0; returns the quantized (noisy in training) attributes of all
    anchors and, with predict_bpp, the sampled bit estimate (Eq. 11)."""
    c = bank.c
    k = bank.n_offsets
    n = anchor.shape[0]
    feat_q = torch.zeros_like(feat)
    scaling_q = torch.zeros_like(scaling)
    offsets_q = torch.zeros_like(offsets)
    already = torch.zeros_like(feat[:, 0]).bool()
    if predict_bpp:
        mean_f, scale_f = torch.zeros_like(feat), torch.zeros_like(feat)
        q_f = torch.zeros_like(feat[:, [0]])
        mean_s, scale_s = torch.zeros_like(scaling), torch.zeros_like(scaling)
        q_s = torch.zeros_like(scaling[:, [0]])
        flat = offsets.view(-1, 3 * k)
        mean_o, scale_o = torch.zeros_like(flat), torch.zeros_like(flat)
        q_o = torch.zeros_like(flat[:, [0]])
    hyper_feat, lik_hyper = bank.prior(hyper, training=training)
    if not use_hyper:  # disable_hyper ablation (official flag)
        hyper_feat = hyper_feat * 0
    if bank.level_scale is None:
        sub = anchor if mask_anchor_bool is None else anchor[mask_anchor_bool]
        bank.level_scale = find_divide_scale(sub.detach(), c["voxel_size"], bank.x_bound_min,
                                             bank.x_bound_max, c["target_ratio"], bank.level_num)
    levels = divide_levels(anchor, c["voxel_size"], bank.level_scale, bank.level_num,
                           mask_anchor_bool)
    _, inverse_list, mapping_list, _ = levels
    content_pre = None
    order = []
    for i in reversed(range(bank.level_num)):
        to_code, orig, feat_in = level_inputs(levels, i, content_pre, hyper_feat, n)
        if to_code.sum() > 0:
            order.append(orig)
            pred = bank.predict(i, feat_in)
            mf, sf, qf = pred["feat"]
            ms, ss, qs = pred["scaling"]
            mo, so, qo = pred["offsets"]
            f_i, s_i, o_i = feat[orig], scaling[orig], offsets[orig]
            if training:
                f_i = f_i + torch.empty_like(f_i).uniform_(-0.5, 0.5) * qf
                s_i = s_i + torch.empty_like(s_i).uniform_(-0.5, 0.5) * qs
                o_i = o_i + torch.empty_like(o_i).uniform_(-0.5, 0.5) * qo.unsqueeze(1)
            else:
                f_i = model.STEMultistep.apply(f_i, qf).detach()
                s_i = model.STEMultistep.apply(s_i, qs).detach()
                o_i = model.STEMultistep.apply(o_i, qo.unsqueeze(1)).detach()
            if predict_bpp:
                mean_f[orig], scale_f[orig], q_f[orig] = mf, sf, qf
                mean_s[orig], scale_s[orig], q_s[orig] = ms, ss, qs
                mean_o[orig], scale_o[orig], q_o[orig] = mo, so, qo
            feat_q[orig] = f_i
            scaling_q[orig] = s_i
            offsets_q[orig] = o_i.reshape(-1, k, 3)
            already[orig] = True
        if i != 0:
            content_pre = extract_context_feat(anchor, feat_q, scaling_q, already, inverse_list,
                                               mapping_list, i)
    if not predict_bpp:
        return feat_q, scaling_q, offsets_q, {"order": order, "levels": levels}
    choose = torch.rand_like(anchor[:, 0]) <= sample
    if mask_anchor_bool is not None:
        choose = choose & mask_anchor_bool
        rate = (mask_anchor_bool.sum() / mask_anchor_bool.numel()).detach()
    else:
        rate = 1
    b_hyper = -torch.log2(lik_hyper[choose])
    b_feat = model.gaussian_bits(feat_q[choose], mean_f[choose], scale_f[choose], q_f[choose],
                                 x_means["feat"])
    b_scal = model.gaussian_bits(scaling_q[choose], mean_s[choose], scale_s[choose],
                                 q_s[choose], x_means["scaling"])
    b_off = model.gaussian_bits(offsets_q[choose].view(-1, 3 * k), mean_o[choose],
                                scale_o[choose], q_o[choose], x_means["offsets"])
    b_off = b_off * masks[choose].repeat(1, 1, 3).view(-1, 3 * k)
    denom = b_feat.numel() + b_scal.numel() + b_off.numel()
    if denom == 0:
        bpp = (feat_q.sum() + scaling_q.sum() + offsets_q.sum() + hyper_feat.sum()) * 0.0
    else:
        bpp = (b_feat.sum() + b_scal.sum() + b_off.sum() + b_hyper.sum()) / denom * rate
    bits = {
        "bit_per_param": bpp,
        "sampled": int(choose.sum()),
        "eligible": int(mask_anchor_bool.sum()) if mask_anchor_bool is not None else n,
        "bit_per_feat_param": b_feat.sum().detach() / max(b_feat.numel(), 1) * rate,
        "bit_per_scaling_param": b_scal.sum().detach() / max(b_scal.numel(), 1) * rate,
        "bit_per_offsets_param": b_off.sum().detach() / max(b_off.numel(), 1) * rate,
        "bit_per_hyper_param": b_hyper.sum().detach() / max(b_hyper.numel(), 1) * rate,
        "order": order,
        "levels": levels,
    }
    return feat_q, scaling_q, offsets_q, bits
