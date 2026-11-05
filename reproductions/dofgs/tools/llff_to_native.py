"""Build native-full/<scene> from LLFF poses_bounds.npy + sparse/0/points3D.bin, for scenes whose
COLMAP cameras.bin/images.bin are unavailable. The LLFF->transforms mapping (c2w columns
perm (1,0,2) signs (1,-1,1), translation unchanged, focal = hwf[2], centred principal point) was
verified to 2e-16 against the real COLMAP conversion on cake and caps."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/sejain/splatify/splatstudio/scripts')
from colmap_to_transforms import read_points, link

scene = sys.argv[1]
raw = Path(f'/home/sejain/datasets/deblurnerf/real_defocus_blur/defocus{scene}')
out = Path(f'/home/sejain/datasets/deblurnerf/native-full/{scene}')
M = np.load('/home/sejain/splatify/reproductions/dofgs/tools/llff_M.npy')
pb = np.load(raw / 'poses_bounds.npy'); P = pb[:, :15].reshape(-1, 3, 5)
names = sorted(p.name for p in (raw / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.png'))
assert len(names) == len(P), (len(names), len(P))
assert not (out.exists() and any(out.iterdir())), 'output must be empty'
(out / 'images').mkdir(parents=True, exist_ok=True)
frames = []
for name, p in zip(names, P):
    src = raw / 'images' / name
    with Image.open(src) as im:
        w, h = im.size
    H, W, f = p[:, 4]
    assert (W, H) == (w, h), (name, W, H, w, h)
    c2w = np.eye(4); c2w[:3, :3] = p[:, :3] @ M; c2w[:3, 3] = p[:, 3]
    link(src, out / 'images' / name)
    frames.append({'file_path': f'images/{name}', 'transform_matrix': c2w.tolist(), 'w': w, 'h': h,
                   'fl_x': float(f), 'fl_y': float(f), 'cx': w / 2, 'cy': h / 2, 'camera_model': 'PINHOLE'})
points, colors = read_points(raw / 'sparse' / '0' / 'points3D.bin')
np.savez(out / 'points.npz', points=points, colors=colors)
meta = {'frames': frames, 'source': {'scene': raw.name, 'images': 'images', 'colmap': 'poses_bounds.npy (LLFF) + sparse/0/points3D.bin; cameras.bin/images.bin unavailable from Drive'}}
(out / 'transforms.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({'output': str(out), 'frames': len(frames), 'points': len(points)}))
