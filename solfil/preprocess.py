"""Image preprocessing shared by all stages: reading, disk-median normalisation, limb-darkening removal, calibration."""
import os

import cv2
import numpy as np

H = W = 2048          # native size of the GONG H-alpha frames; all masks are produced at this size
S1 = 1024             # half resolution used by most networks

_GRID = {}


def list_images(folder):
    """Sorted image stems of a folder and a dict stem -> path."""
    paths = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    return sorted(paths), paths


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def small(a, interp=cv2.INTER_AREA):
    """2048 px -> 1024 px."""
    return cv2.resize(a, (S1, S1), interpolation=interp)


def up(a):
    """1024 px map -> 2048 px (bilinear)."""
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)


def disk_median(a):
    """Median brightness of the solar disk (pixels > 20); 128 for an (almost) empty frame."""
    d = np.asarray(a)
    d = d[d > 20]
    return float(np.median(d)) if d.size > 1000 else 128.


def norm(x, med):
    """Image divided by its disk median, scaled so that the median maps to 0.5, clipped to [0, 1.5]."""
    return np.clip(x.astype(np.float32) / max(med, 1.) * 0.5, 0, 1.5)


def flatten(im):
    """Image divided by its radial median profile (limb-darkening removal), mapped [0.4, 1.4] -> [0, 255].

    The disk centre and radius are estimated from the image itself (pixels > 20 of a 512 px copy); the radial
    profile is the median in 120 rings. Faint filaments near the limb get the same contrast as at disk centre.
    """
    if im.shape not in _GRID:
        _GRID[im.shape] = np.indices(im.shape, dtype=np.float32)
    YY, XX = _GRID[im.shape]
    a = im.astype(np.float32)
    sm = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
    m = sm > 20
    if m.sum() < 1000:
        return np.zeros(im.shape, np.uint8)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() * 4 + 1.5, xs.mean() * 4 + 1.5
    R = np.sqrt(m.sum() / np.pi) * 4
    yy, xx = np.indices(sm.shape)
    rs = np.hypot(yy * 4 + 1.5 - cy, xx * 4 + 1.5 - cx) / R
    nb = 120
    b = np.clip((rs * nb).astype(int), 0, nb + 9)
    prof = np.full(nb + 10, np.nan, np.float32)
    for k in range(nb + 10):
        v = sm[(b == k) & m]
        if v.size > 20:
            prof[k] = np.median(v)
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(nb + 10), np.where(ok)[0], prof[ok])
    r = np.hypot(YY - cy, XX - cx) / R
    pr = np.interp(r * nb, np.arange(nb + 10), prof)
    f = np.where((r < 1.0) & (a > 20), a / np.maximum(pr, 1.), 0.)
    return (np.clip((f - 0.4), 0, 1.) * 255).astype(np.uint8)


def lut_from_hist(h_ref, h_new):
    """Lookup table that maps a value distribution (256-bin histogram h_new) onto the reference histogram h_ref."""
    h_ref = np.asarray(h_ref, float)
    h_new = np.asarray(h_new, float)
    c_ref = np.cumsum(h_ref) / h_ref.sum()
    c_new = np.cumsum(h_new) / h_new.sum()
    return np.clip(np.searchsorted(c_ref, c_new, side='left'), 0, 255).astype(np.uint8)


def quantile_match(h_ref, maps):
    """Quantile-match a stack of uint8 maps to the out-of-fold value distribution.

    The decision layer was trained on out-of-fold maps (one model per image); the test maps are the mean of five fold
    models and therefore smoother. Matching their value distribution to the out-of-fold one keeps the thresholds and
    descriptors on the same scale.
    """
    lut = lut_from_hist(h_ref, np.bincount(maps.ravel(), minlength=256))
    return lut[maps]


def mean_u8(stack):
    """Rounded uint8 mean of a list of uint8 maps."""
    acc = np.zeros(stack[0].shape, np.uint16)
    for a in stack:
        acc += a
    return ((acc + len(stack) // 2) // len(stack)).astype(np.uint8)


def spread(stack):
    """Mean absolute deviation of the members from their mean (uint8): where the networks disagree."""
    a = np.stack(stack).astype(np.float32)
    return np.abs(a - a.mean(0)).mean(0).round().astype(np.uint8)
