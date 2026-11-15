"""Figures of the intermediate results of `solfil.pipeline.run` (read from its cache folder)."""
import json
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

H = W = 2048


def _dec(c):
    return mu.decode(dict(size=[H, W], counts=c.encode() if isinstance(c, str) else c)).astype(bool)


def _outline(rgb, m, color, th=5):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, color, th)
    return rgb


def _gray(a):
    return np.repeat(np.asarray(a)[..., None], 3, 2).copy()


def load(cache, sub_csv):
    stems = json.load(open(f'{cache}/meta.json'))['stems']
    sub = pd.read_csv(sub_csv)
    sub['stem'] = sub.filament_id.str.rsplit('_', n=1).str[0]
    return dict(stems=stems, sub=sub, img=np.load(f'{cache}/img2k.npy', mmap_mode='r'), flat=np.load(f'{cache}/flat2k.npy', mmap_mode='r'),
                prob=np.load(f'{cache}/prob2k.npy', mmap_mode='r'), spine=np.load(f'{cache}/spine1k.npy', mmap_mode='r'),
                std=np.load(f'{cache}/std1k.npy', mmap_mode='r'), st1=pickle.load(open(f'{cache}/stage1.pkl', 'rb')),
                det=pickle.load(open(f'{cache}/m34.pkl', 'rb')), cache=cache)


def stages_figure(R, s, lam_best, path):
    """One test image through the pipeline: input, flattened input, ensemble map, spine map, simulated annotators,
    detector instances, first-stage selection, final instances."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    i = R['stems'].index(s)
    im = np.asarray(R['img'][i])
    from .pipeline import sim_levels
    lev = sim_levels(np.load(f"{R['cache']}/E_lut.npy"), np.load(f"{R['cache']}/E_pa8.npy", mmap_mode='r')[i])
    det = [d for d in R['det'][0][s][0] if d[1] >= 0.5]
    rgb_det = _gray(im)
    for c, _ in det:
        _outline(rgb_det, _dec(c), (255, 140, 0))
    rgb_sel = _gray(im)
    for c, q in R['st1']['sel'][s]:
        if q > lam_best:
            _outline(rgb_sel, _dec(c), (0, 160, 255))
    rgb_fin = _gray(im)
    for c in R['sub'][R['sub'].stem == s].segmentation_rle:
        _outline(rgb_fin, _dec(c), (255, 40, 40))
    panels = [('(a) input image', im, 'gray'), ('(b) limb-flattened image', np.asarray(R['flat'][i]), 'gray'),
              ('(c) calibrated ensemble map B+C+D', np.asarray(R['prob'][i]), 'magma'), ('(d) spine map (member D)', np.asarray(R['spine'][i]), 'magma'),
              ('(e) simulated annotators drawing a pixel', (lev >= 2).sum(0), 'viridis'), ('(f) Mask R-CNN instances (score >= 0.5)', rgb_det, None),
              (f'(g) first-stage selection (q > {lam_best:.2f})', rgb_sel, None), ('(h) final instances', rgb_fin, None)]
    fig, ax = plt.subplots(2, 4, figsize=(20, 10.4))
    for a, (t, x, cm) in zip(ax.ravel(), panels):
        a.imshow(x, cmap=cm) if cm else a.imshow(x)
        a.set_title(t, fontsize=13); a.axis('off')
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def examples_figure(R, stems, path, crop=None):
    """Final instances (red outlines) on several test images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(stems), figsize=(5.2 * len(stems), 5.4))
    for a, s in zip(np.atleast_1d(ax), stems):
        i = R['stems'].index(s)
        rgb = _gray(np.asarray(R['img'][i]))
        g = R['sub'][R['sub'].stem == s]
        for k, c in enumerate(g.segmentation_rle):
            _outline(rgb, _dec(c), (255, 40, 40), 6)
        a.imshow(rgb if crop is None else rgb[crop[0]:crop[1], crop[2]:crop[3]]); a.axis('off')
        a.set_title(f'{s}  ({len(g)} filaments)', fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def refine_figure(R, refined, path, n=4):
    """Oriented crops of selected instances: image, seed, refined probability, for the n first test instances
    with at least 4000 px."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .refine import build_crops
    rows, k0 = [], 0
    for i, s in enumerate(R['stems']):
        sel = R['st1']['sel'][s]
        big = [k for k, (c, q) in enumerate(sel) if _dec(c).sum() >= 4000]
        if big:
            crops, _ = build_crops([sel[big[0]]], np.asarray(R['img'][i]), np.asarray(R['prob'][i]), np.asarray(R['spine'][i]))
            rows.append((s, crops[0], refined[k0 + big[0]], sel[big[0]][1]))
        k0 += len(sel)
        if len(rows) == n:
            break
    fig, ax = plt.subplots(len(rows), 4, figsize=(16, 1.6 * len(rows) + 0.6))
    for r, (s, x, rp, q) in enumerate(rows):
        for c, (t, a) in enumerate((('image', x[0]), ('ensemble map', x[2]), ('seed (first stage)', x[4]), ('refined probability', rp))):
            ax[r, c].imshow(a, cmap='gray' if c == 0 else 'magma', vmin=0, vmax=255); ax[r, c].axis('off')
            if r == 0:
                ax[r, c].set_title(t, fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
