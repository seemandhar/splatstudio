"""2D Haar DWT (paper Eq. 2, §V-A "we used the Haar wavelet"); default normalisation "average".

For each 2x2 block [[a, b], [c, d]] (rows = image y):
    LL = (a + b + c + d) / k      low-pass along rows and columns
    LH = (a + b - c - d) / k      high-pass along columns (y), low-pass along rows (x)
    HL = (a - b + c - d) / k      low-pass along y, high-pass along x
    HH = (a - b - c + d) / k      high-pass in both directions (diagonal detail)
with divisor k; k = 2 is the orthonormal (pywt / pytorch_wavelets) form.
Default norm="average", k = 4: LL = 2x2 block mean, i.e. "in the same space as the original image,
only a coarser version" (§IV-B; Figs. 2-3 show LL at image brightness, only HH contrast-enhanced).
norm="orthonormal" (k = 2) is a labelled option only. An odd height/width is zero-padded at the
end (pytorch_wavelets mode='zero'). Multi-level DWT repeats the transform on LL.
"""

import torch.nn.functional as F

DIVISORS = {"orthonormal": 2.0, "average": 4.0}


def haar_dwt(x, norm="average"):
    """x [..., H, W] -> (LL, LH, HL, HH), each [..., ceil(H/2), ceil(W/2)]."""
    k = DIVISORS[norm]
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2))
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return (a + b + c + d) / k, (a + b - c - d) / k, (a - b + c - d) / k, (a - b - c + d) / k


def ll_pyramid(x, levels, norm="average"):
    """[X^1_LL, ..., X^N_LL] of x [..., H, W]."""
    out = []
    for _ in range(int(levels)):
        x = haar_dwt(x, norm)[0]
        out.append(x)
    return out
