"""Official LLFF near/far bounds for the 3 FSGS training views of each scene, copied verbatim
from the FSGS port (reproductions/fsgs/plugin/ss_fsgs/density.py, which reads them from
~/datasets/llff/fewview3-r8/<scene>/prep.json = poses_bounds.npy). generate_random_poses_llff
only uses min(near) and max(far)."""

import numpy as np

# LLFF poses_bounds.npy near/far of the 3 FSGS training views (min near / max far over the views,
# the only quantities generate_random_poses_llff uses), from ~/datasets/llff/fewview3-r8/<s>/
# prep.json; matched to the scene by the training-camera centres (c2w translation, world units).
SCENE_BOUNDS = {
    "fern": {"near": 17.542520113884287, "far": 78.71925302875819, "train_centers": [
        [-1.658869, -1.609145, -0.242867], [-4.766545, 0.720178, -0.17121],
        [-3.881079, 1.736659, 0.7926]]},
    "flower": {"near": 22.656267550734345, "far": 162.94281555616587, "train_centers": [
        [-4.323173, -2.258927, 1.372147], [-4.035961, 0.720233, -0.023411],
        [-4.668914, 3.036076, -1.819622]]},
    "fortress": {"near": 13.472437825125102, "far": 128.38342286860328, "train_centers": [
        [-2.343452, -2.985312, 1.402754], [-0.596759, 0.169303, -0.406764],
        [4.019429, 2.461339, 0.372861]]},
    "horns": {"near": 12.473701480698228, "far": 168.81094102559487, "train_centers": [
        [-3.77684, -3.147624, -1.43279], [3.351668, 0.198719, 0.574782],
        [3.00166, 2.723918, 0.929056]]},
    "leaves": {"near": 35.262850166998774, "far": 449.9130266665557, "train_centers": [
        [-3.326046, -2.317046, -1.206822], [-1.263424, 0.250338, 0.229431],
        [6.435536, 2.427958, 0.479507]]},
    "orchids": {"near": 15.062411908163723, "far": 96.9429995809968, "train_centers": [
        [-1.975985, -3.74603, -0.495709], [0.042977, 0.292588, -0.050057],
        [0.474495, 3.573621, 0.663743]]},
    "room": {"near": 10.729799505777695, "far": 48.929631822500625, "train_centers": [
        [-3.619974, -3.791045, 0.452161], [0.16049, 2.171683, -0.405748],
        [-0.337692, 1.516422, 0.225718]]},
    "trex": {"near": 26.506352452490084, "far": 112.93942182214286, "train_centers": [
        [-1.90164, -2.832524, 0.605863], [-0.239929, 1.547036, -0.79153],
        [3.713924, 1.138788, -0.674476]]},
}


def official_bounds(table, viewmats, tol=1e-3):
    """Return (scene, [[near, far]]) whose recorded training-camera centres match these views."""
    centers = np.stack([np.linalg.inv(v)[:3, 3] for v in viewmats])
    for scene, entry in table.items():
        ref = np.asarray(entry["train_centers"], dtype=np.float64)
        if ref.shape == centers.shape and np.abs(ref - centers).max() < tol:
            return scene, np.array([[float(entry["near"]), float(entry["far"])]])
    raise ValueError("pseudo_bounds='official': no scene_bounds entry matches the training "
                     "cameras (add the scene's poses_bounds near/far or set pseudo_bounds='points')")

