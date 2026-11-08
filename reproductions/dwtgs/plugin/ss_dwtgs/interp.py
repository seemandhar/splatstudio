"""Novel views by spherical interpolation between training views (paper Fig. 2, note 3:
"Novel view sampling is performed using spherical interpolation between training views, only at
every K iterations"). The interpolation details are unstated; this is the simplest literal
reading: pick a random unordered pair of distinct training cameras and t ~ U(0, 1), slerp the
camera-to-world rotations, lerp the camera centres (and intrinsics; LLFF views share them).
"""

import numpy as np


def rot_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), Shepperd's method."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:  # shortest arc
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def interpolate_viewmat(w2c_a, w2c_b, t):
    """OpenCV world-to-camera matrices -> interpolated world-to-camera matrix."""
    ca, cb = np.linalg.inv(w2c_a), np.linalg.inv(w2c_b)
    R = quat_to_rot(slerp(rot_to_quat(ca[:3, :3]), rot_to_quat(cb[:3, :3]), t))
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = (1 - t) * ca[:3, 3] + t * cb[:3, 3]
    return np.linalg.inv(c2w)
