"""Poisson meshing of level-set points (sugar_extractors/coarse_mesh.py:300-470) with Open3D.

Open3D is not part of the splatstudio env; it lives in a separate target directory
(default ~/models/sugar-pylib) that is appended (not prepended) to sys.path on first use.
"""

import os
import sys

import numpy as np


def import_open3d(path="~/models/sugar-pylib"):
    try:
        import open3d as o3d
    except ImportError:
        path = os.path.expanduser(path)
        if path not in sys.path:
            sys.path.append(path)
        import open3d as o3d
    return o3d


def _clean(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh


def _poisson(o3d, points, normals, colors, depth, quantile, target, threads):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,
                                                                     n_threads=threads)
    dens = np.asarray(dens)
    if quantile > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, quantile))
    return _clean(mesh.simplify_quadric_decimation(int(target)))


def poisson_mesh(points, normals, colors, center, extent, *, depth=10, quantile=0.1,
                 target=1_000_000, fg_factor=1.0, bg_factor=4.0, project=True,
                 open3d_path="~/models/sugar-pylib", threads=8):
    """Returns (verts [V,3], faces [F,3], vertex colours [V,3]) as numpy arrays.

    fg = inside the +-E box around the mean camera centre, bg = inside +-4E and not fg
    (center_bbox=True). Each part: outlier removal (20, 20), Poisson, drop the 10 % lowest
    density vertices, quadric decimation to `target` triangles, cleanup. Merged vertices are
    then snapped to their nearest level-set point (project_mesh_on_surface_points).
    """
    from scipy.spatial import cKDTree

    o3d = import_open3d(open3d_path)
    rel = points - center[None]
    fg = (np.abs(rel) < fg_factor * extent).all(axis=1)
    bg = (np.abs(rel).max(axis=1) < bg_factor * extent) & ~fg
    parts = []
    for mask in (fg, bg):
        if mask.sum() > 0:
            parts.append(_poisson(o3d, points[mask], normals[mask], colors[mask], depth, quantile,
                                  target, threads))
    if not parts:
        raise RuntimeError("SuGaR mesh extraction found no level-set points")
    mesh = parts[0]
    for extra in parts[1:]:
        mesh = mesh + extra
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if project:
        _, nn = cKDTree(points).query(verts, k=1, workers=threads)
        verts, vcol = points[nn], colors[nn]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(vcol.astype(np.float64))
        _clean(mesh)
    return (np.asarray(mesh.vertices).astype(np.float32), np.asarray(mesh.triangles).astype(np.int64),
            np.asarray(mesh.vertex_colors).astype(np.float32))
