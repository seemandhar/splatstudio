"""Billboard texture fields and their official activations / initial values / 8-bit storage.

Official code: scene/gaussian_model.py (``create_from_pcd``, ``compress_texture``,
``decompress_texture``) and ``assets/alpha_init_gaussian_small.png`` (16x16, copied next to this
file; its pixel values are reproduced below and checked by the unit tests).
"""

import torch

ALPHA = "bb_alpha"  # [N,16,16] logits of T^alpha
COLOR = "bb_color"  # [N,3,16,16] logits of T^RGB
SIZE = 16  # S_T (paper §4.1, Supp. Tab. 8)
LOGIT_CAP = 1e4  # official logit(0)/logit(1) = -inf/+inf; stored as -/+1e4 (sigmoid identical)

# cv2.imread("assets/alpha_init_gaussian_small.png")[..., 0] (uint8)
GAUSSIAN_PNG = (
    (0, 0, 0, 1, 2, 3, 3, 4, 4, 4, 3, 2, 1, 1, 0, 0),
    (0, 0, 1, 2, 4, 6, 9, 10, 11, 10, 7, 5, 3, 2, 1, 0),
    (0, 1, 3, 5, 10, 15, 21, 25, 25, 23, 18, 12, 7, 4, 2, 1),
    (1, 2, 5, 11, 20, 32, 43, 51, 53, 48, 37, 26, 15, 8, 3, 1),
    (2, 4, 10, 20, 36, 57, 77, 92, 96, 86, 67, 46, 27, 14, 7, 3),
    (3, 6, 15, 32, 57, 89, 121, 144, 149, 135, 106, 72, 43, 22, 10, 4),
    (3, 9, 21, 43, 77, 121, 166, 197, 204, 184, 144, 99, 59, 31, 13, 5),
    (4, 10, 25, 51, 92, 144, 197, 234, 242, 219, 171, 117, 70, 36, 16, 7),
    (4, 11, 25, 53, 96, 149, 204, 242, 251, 226, 177, 121, 72, 37, 17, 7),
    (4, 10, 23, 48, 86, 135, 184, 219, 226, 204, 160, 110, 65, 34, 15, 6),
    (3, 7, 18, 37, 67, 106, 144, 171, 177, 160, 126, 86, 51, 26, 12, 5),
    (2, 5, 12, 26, 46, 72, 99, 117, 121, 110, 86, 59, 35, 18, 8, 3),
    (1, 3, 7, 15, 27, 43, 59, 70, 72, 65, 51, 35, 21, 11, 5, 2),
    (1, 2, 4, 8, 14, 22, 31, 36, 37, 34, 26, 18, 11, 6, 2, 1),
    (0, 1, 2, 3, 7, 10, 13, 16, 17, 15, 12, 8, 5, 2, 1, 0),
    (0, 0, 1, 1, 3, 4, 5, 7, 7, 6, 5, 3, 2, 1, 0, 0),
)
INIT_COLOR = 0.1  # texture_color = ones * 0.1 before inverse sigmoid
SH_DC_OFFSET = 0.1  # features[:, :3, 0] = RGB2SH(colour) - 0.1


def gaussian_map(device=None):
    """G = PNG / 255 (``self._texture_alpha_init``), [16,16] float32."""
    return torch.tensor(GAUSSIAN_PNG, dtype=torch.float32, device=device) / 255.0


def initial_alpha(device=None):
    """Initial T^alpha value 0.5 G (``inverse_sigmoid(0.5 * texture_alpha)`` then sigmoid); also
    the A0 of the alpha regulariser (train.py ``initial_texture_alpha``)."""
    return 0.5 * gaussian_map(device)


def safe_logit(a):
    """utils.general_utils.inverse_sigmoid = log(a/(1-a)) with +-inf mapped to +-LOGIT_CAP."""
    out = torch.log(a / (1 - a))
    out = torch.where(a <= 0, torch.full_like(a, -LOGIT_CAP), out)
    return torch.where(a >= 1, torch.full_like(a, LOGIT_CAP), out)


def initial_logits(n, device=None):
    alpha = safe_logit(initial_alpha(device)).expand(n, SIZE, SIZE).contiguous()
    color = torch.full((n, 3, SIZE, SIZE), INIT_COLOR, device=device)
    return alpha, safe_logit(color)


def activated(params):
    return params[ALPHA].sigmoid(), params[COLOR].sigmoid()


def quantize(alpha, color):
    """save_texture -> load_texture round trip used by the official render.py:
    alpha stored as round(((a - G) + 1) / 2 * 255), restored as clip(q/255*2 - 1 + G, 0, 1);
    colour stored as round(c * 255) / 255. Inputs/outputs are activated textures."""
    g = gaussian_map(alpha.device)
    qa = torch.round(((alpha - g) + 1) * 0.5 * 255) / 255
    qa = (qa * 2 - 1 + g).clamp(0, 1)
    qc = torch.round(color * 255) / 255
    return qa, qc
