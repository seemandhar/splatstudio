"""Official CoherentGS LLFF preprocessing functions, vendored verbatim for exactness.

* sparse_bilateral_filtering / vis_depth_discontinuity / bilateral_filter / rolling_window:
  CoherentGS bilateral_filtering.py (originally from 3D-Photo-Inpainting, Shih et al. 2020).
* get_depth_bins: CoherentGS scene/depth_layering.py (Copyright 2022 Google LLC, Apache-2.0;
  from 3D Moments, Wang et al. 2022).
* CONFIG: the keys of the official argument.yaml that these functions read.
Only unused timing variables were dropped; numerics are unchanged.
"""

from functools import reduce

import numpy as np
import torch

CONFIG = {"filter_size": [7, 7, 5, 5, 5], "sigma_s": 4.0, "sigma_r": 0.5,
          "depth_threshold": 0.1, "sparse_iter": 5}

def sparse_bilateral_filtering(
    depth, image, config, HR=False, mask=None, gsHR=True, edge_id=None, num_iter=None, num_gs_iter=None, spdb=False
):
    """
    config:
    - filter_size
    """

    save_images = []
    save_depths = []
    save_discontinuities = []
    vis_depth = depth.copy()
    backup_vis_depth = vis_depth.copy()

    depth_max = vis_depth.max()
    depth_min = vis_depth.min()
    vis_image = image.copy()
    for i in range(num_iter):
        if isinstance(config["filter_size"], list):
            window_size = config["filter_size"][i]
        else:
            window_size = config["filter_size"]
        vis_image = image.copy()
        save_images.append(vis_image)
        save_depths.append(vis_depth)
        u_over, b_over, l_over, r_over = vis_depth_discontinuity(vis_depth, config, mask=mask)
        vis_image[u_over > 0] = np.array([0, 0, 0])
        vis_image[b_over > 0] = np.array([0, 0, 0])
        vis_image[l_over > 0] = np.array([0, 0, 0])
        vis_image[r_over > 0] = np.array([0, 0, 0])

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1
        save_discontinuities.append(discontinuity_map)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter(
            vis_depth, config, discontinuity_map=discontinuity_map, HR=HR, mask=mask, window_size=window_size
        )

    return save_images, save_depths


def vis_depth_discontinuity(depth, config, vis_diff=False, label=False, mask=None):
    """
    config:
    -
    """
    if label == False:
        disp = 1./depth
        u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > config['depth_threshold']).astype(np.float32)
        b_over = (np.abs(b_diff) > config['depth_threshold']).astype(np.float32)
        l_over = (np.abs(l_diff) > config['depth_threshold']).astype(np.float32)
        r_over = (np.abs(r_diff) > config['depth_threshold']).astype(np.float32)
    else:
        disp = depth
        u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > 0).astype(np.float32)
        b_over = (np.abs(b_diff) > 0).astype(np.float32)
        l_over = (np.abs(l_diff) > 0).astype(np.float32)
        r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode='constant')
    b_over = np.pad(b_over, 1, mode='constant')
    l_over = np.pad(l_over, 1, mode='constant')
    r_over = np.pad(r_over, 1, mode='constant')
    u_diff = np.pad(u_diff, 1, mode='constant')
    b_diff = np.pad(b_diff, 1, mode='constant')
    l_diff = np.pad(l_diff, 1, mode='constant')
    r_diff = np.pad(r_diff, 1, mode='constant')

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]

def bilateral_filter(depth, config, discontinuity_map=None, HR=False, mask=None, window_size=False):
    sort_time = 0
    replace_time = 0
    filter_time = 0
    init_time = 0
    filtering_time = 0
    sigma_s = config['sigma_s']
    sigma_r = config['sigma_r']
    if window_size == False:
        window_size = config['filter_size']
    midpt = window_size//2
    ax = np.arange(-midpt, midpt+1.)
    xx, yy = np.meshgrid(ax, ax)
    if discontinuity_map is not None:
        spatial_term = np.exp(-(xx**2 + yy**2) / (2. * sigma_s**2))

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1,1), (1,1)), 'edge')
    pad_depth = np.pad(depth, (midpt,midpt), 'edge')
    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1,1), (1,1)), 'edge')
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt,midpt), 'edge')
        pad_discontinuity_hole = 1 - pad_discontinuity_map
    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1,1])
    if discontinuity_map is not None:
        pad_discontinuity_patches = rolling_window(pad_discontinuity_map, [window_size, window_size], [1,1])
        pad_discontinuity_hole_patches = rolling_window(pad_discontinuity_hole, [window_size, window_size], [1,1])

    if mask is not None:
        pad_mask = np.pad(mask, (midpt,midpt), 'constant')
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1,1])
    from itertools import product
    if discontinuity_map is not None:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue
                if discontinuity_map is not None:
                    if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]
                if discontinuity_map is not None:
                    coef = discontinuity_holes.astype(np.float32)
                    if mask is not None:
                        coef = coef * pad_mask_patches[pi, pj]
                else:
                    range_term = np.exp(-(depth_patch-patch_midpt)**2 / (2. * sigma_r**2))
                    coef = spatial_term * range_term
                if coef.max() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.max() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef/(coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if discontinuity_map is not None:
                    if pad_discontinuity_patches[pi, pj][window_size//2, window_size//2] == 1:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = (1. - discontinuity_patch)
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]
                range_term = np.exp(-(depth_patch-patch_midpt)**2 / (2. * sigma_r**2))
                if discontinuity_map is not None:
                    coef = spatial_term * range_term * discontinuity_holes
                else:
                    coef = spatial_term * range_term
                if coef.sum() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.sum() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef/(coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output

def rolling_window(a, window, strides):
    assert len(a.shape)==len(window)==len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"
    shape_fn = lambda i,w,s: (a.shape[i]-w)//s + 1
    shape = [shape_fn(i,w,s) for i,(w,s) in enumerate(zip(window, strides))] + list(window)
    def acc_shape(i):
        if i+1>=len(a.shape):
            return 1
        else:
            return reduce(lambda x,y:x*y, a.shape[i+1:])
    _strides = [acc_shape(i)*s*a.itemsize for i,s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)


def get_depth_bins(depth=None, disparity=None, num_bins=None):
    """
    :param depth: [1, 1, H, W]
    :param disparity: [1, 1, H, W]
    :return: depth_bins
    """

    assert (disparity is not None) or (depth is not None)
    if disparity is None:
        assert depth.min() > 1e-2
        disparity = 1. / depth

    if depth is None:
        depth = 1. / torch.clamp(disparity, min=1e-2)

    assert depth.shape[:2] == (1, 1) and disparity.shape[:2] == (1, 1)
    disparity_max = disparity.max().item()
    disparity_min = disparity.min().item()
    disparity_feat = disparity[:, :, ::10, ::10].reshape(-1, 1).cpu().numpy()
    disparity_feat = (disparity_feat - disparity_min) / (disparity_max - disparity_min)
    if num_bins is None:
        n_clusters = None
        distance_threshold = 5
    else:
        n_clusters = num_bins
        distance_threshold = None
    from sklearn.cluster import AgglomerativeClustering  # PYTHONPATH=~/models/coherentgs/pylib

    result = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold).fit(disparity_feat)
    num_bins = result.n_clusters_ if n_clusters is None else n_clusters
    depth_bins = [depth.min().item()]
    for i in range(num_bins):
        th = (disparity_feat[result.labels_ == i]).min()
        th = th * (disparity_max - disparity_min) + disparity_min
        depth_bins.append(1. / th)

    depth_bins = sorted(depth_bins)
    depth_bins[0] = depth.min().numpy() - 1e-6
    depth_bins[-1] = depth.max().numpy() + 1e-6
    return depth_bins
