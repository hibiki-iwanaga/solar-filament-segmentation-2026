"""Stage-2 refinement (stage 15) and the second quality model (stage 16).

Every instance selected by the first quality model is cut out in an oriented crop (principal axis along x, 384 x 128
px, at most 2x magnification) with six channels: image / disk median, flattened image, calibrated ensemble map,
calibrated spine map, the instance itself (the seed) and the other selected instances. Three refiner networks
redraw the outline; their mean probability is pasted back into the image in decreasing predicted quality, so the
final instances never overlap. The refined outline is described again (how it relates to its seed and to the
refined probability) and a second LightGBM predicts its matched IoU; the mean of the first- and second-stage
predictions orders and filters the final instances.
"""
import numpy as np
import cv2
from pycocotools import mask as mu

from .preprocess import flatten

H = W = 2048
CW, CH = 384, 128
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def frame(seed):
    """Affine map image (2048 px) -> crop: principal axis along x, instance + margin fitted, at most 2x."""
    ys, xs = np.nonzero(seed)
    cx, cy = xs.mean(), ys.mean()
    sxx, syy, sxy = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean(), ((xs - cx) * (ys - cy)).mean()
    phi = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    c, sn = np.cos(phi), np.sin(phi)
    u = (xs - cx) * c + (ys - cy) * sn; v = -(xs - cx) * sn + (ys - cy) * c
    um, vm = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    cx, cy = cx + um * c - vm * sn, cy + um * sn + vm * c          # centre of the oriented box
    L, Wd = u.max() - u.min() + 1, v.max() - v.min() + 1
    sc = min(CW / (L * 1.2 + 48), CH / (Wd * 2.0 + 48), 2.0)
    M = np.array([[sc * c, sc * sn, 0.], [-sc * sn, sc * c, 0.]])
    M[:, 2] = np.array([CW / 2, CH / 2]) - M[:, :2] @ np.array([cx, cy])
    return M


def warp(a, M, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(a, M, (CW, CH), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def build_crops(sel, im, prob, spine):
    """Crops of the selected instances of one image.

    sel: [(RLE counts, quality)]; im: image (uint8, 2048 px); prob: calibrated ensemble map (2048 px);
    spine: calibrated spine map (1024 px). Returns (crops [n, 6, 128, 384] uint8, affine maps [n]).
    """
    d = im[im > 20]; med = float(np.median(d)) if d.size > 1000 else 128.
    raw = np.clip(im.astype(np.float32) / med * 128, 0, 255).astype(np.uint8)
    fl = flatten(im)
    seeds = [mu.decode(dict(size=[H, W], counts=c.encode())) for c, _ in sel]
    allm = np.zeros((H, W), np.uint8)
    for k, m in enumerate(seeds):
        allm[m > 0] = k + 1
    xs, Ms = [], []
    for k, seed in enumerate(seeds):
        M = frame(seed)
        M1 = M.copy(); M1[:, :2] *= 2.         # maps at 1024 px: image x = 2 x_1024
        x = np.stack([warp(raw, M), warp(fl, M), warp(prob, M if prob.shape[0] == H else M1),
                      warp(spine, M1) if spine is not None else np.zeros((CH, CW), np.uint8),
                      warp(seed * 255, M), warp(((allm > 0) & (allm != k + 1)).astype(np.uint8) * 255, M)])
        xs.append(x); Ms.append(M)
    return (np.stack(xs) if xs else np.zeros((0, 6, CH, CW), np.uint8)), Ms


def paste(pred, M, t):
    """Refined crop probability -> image frame: (mask >= t, probability, x0, y0) inside the bounding box of the crop."""
    Mi = cv2.invertAffineTransform(M)
    cs = np.array([[0, 0, 1], [CW, 0, 1], [0, CH, 1], [CW, CH, 1]], np.float64) @ Mi.T
    x0, y0 = np.clip(np.floor(cs.min(0)).astype(int), 0, W); x1, y1 = np.clip(np.ceil(cs.max(0)).astype(int) + 1, 0, W)
    Mb = M.copy(); Mb[:, 2] += M[:, :2] @ np.array([x0, y0], np.float64)
    pb = cv2.warpAffine(pred, Mb, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return pb >= t * 255, pb, x0, y0


def refine_seq(sel, preds, Ms, t, collapse=0.5, use=None):
    """Pixel-disjoint refined outlines of one image in decreasing predicted quality.

    Only the connected parts of the refined mask that touch the seed are kept; an instance whose refined outline
    collapses to less than half of its seed is dropped. Returns per instance (rle or None, mean refined probability).
    """
    order = np.argsort([-u for _, u in sel], kind='stable')
    taken = np.zeros((H, W), bool); full = np.zeros((H, W), np.uint8, order='F')
    res = [(None, 0.)] * len(sel)
    for k in order:
        if use is not None and not use[k]:
            continue
        seed = mu.decode(dict(size=[H, W], counts=sel[k][0].encode())).astype(bool)
        pb, pv, x0, y0 = paste(preds[k], Ms[k], t)
        sb = seed[y0:y0 + pb.shape[0], x0:x0 + pb.shape[1]]
        n, lab = cv2.connectedComponents(pb.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[pb & sb]); keep = keep[keep > 0]
        m = np.isin(lab, keep) if len(keep) else np.zeros(pb.shape, bool)
        if m.sum() < collapse * seed.sum():
            continue
        pm = float(pv[m].mean()) / 255.
        h, w = m.shape
        m &= ~taken[y0:y0 + h, x0:x0 + w]
        if m.sum() < 50:
            continue
        taken[y0:y0 + h, x0:x0 + w] |= m
        full[y0:y0 + h, x0:x0 + w] = m
        res[k] = (mu.encode(full), pm)
        full[y0:y0 + h, x0:x0 + w] = 0
    return res


def describe_refined(sel, preds, Ms, cand_feats, td=0.40):
    """Descriptors of the (not yet disjoint) refined outline of every selected instance of one image.

    cand_feats: {RLE counts: first-stage descriptor dict} of the candidates of the image.
    """
    us = np.array([q for _, q in sel]); rk = np.argsort(np.argsort(-us))
    out = []; full = np.zeros((H, W), np.uint8, order='F'); rles = []
    for k, (c, u) in enumerate(sel):
        seed = mu.decode(dict(size=[H, W], counts=c.encode())).astype(bool)
        pb, pv, x0, y0 = paste(preds[k], Ms[k], td)
        sb = seed[y0:y0 + pb.shape[0], x0:x0 + pb.shape[1]]
        n, lab = cv2.connectedComponents(pb.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[pb & sb]); keep = keep[keep > 0]
        m = np.isin(lab, keep) if len(keep) else np.zeros(pb.shape, bool)
        a_s, a_m = float(seed.sum()), float(m.sum())
        f = dict(cand_feats.get(c, {}))
        f.update(u=u, u_rank=int(rk[k]), n_sel=len(sel), found=int(c in cand_feats), sc=float(np.sqrt(abs(np.linalg.det(Ms[k][:, :2])))),
                 r_area=np.log1p(a_m), r_ratio=np.log((a_m + 50) / (a_s + 50)), r_ncomp=int(len(keep)), r_nall=int(n - 1),
                 crop_max=float(preds[k].max()) / 255., crop_mean=float(preds[k].mean()) / 255.)
        if a_m > 0:
            pin = pv[m].astype(np.float32) / 255.
            inter = float((m & sb).sum())
            ring = (cv2.dilate(m.astype(np.uint8), K7) > 0) & ~m
            lo = float((pv >= 0.25 * 255).sum()); hi = float((pv >= 0.7 * 255).sum())
            f.update(r_pm=pin.mean(), r_p10=np.percentile(pin, 10), r_p50=np.median(pin), r_p90=np.percentile(pin, 90),
                     r_iou=inter / (a_m + a_s - inter), r_kept=inter / max(a_s, 1), r_ring=float(pv[ring].mean()) / 255. if ring.any() else 0.,
                     r_soft=np.log((lo + 50) / (hi + 50)), r_hi=hi / a_m, r_out=float((pb & ~m).sum()) / a_m)
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            r = mu.encode(full); full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = 0
        else:
            r = None
        rles.append(r); out.append(f)
    ok = [i for i, r in enumerate(rles) if r is not None]
    if len(ok) > 1:
        ov = np.array(mu.iou([rles[i] for i in ok], [rles[i] for i in ok], [0] * len(ok)))
        for a, i in enumerate(ok):
            hi_ = [b for b, j in enumerate(ok) if us[j] > us[i]]
            out[i]['r_ovmax'] = float(np.delete(ov[a], a).max()); out[i]['r_ovhi'] = float(ov[a, hi_].max()) if hi_ else 0.
    return out


def final_instances(sel, q, preds, Ms, t, lam, pmin):
    """Final pixel-disjoint instances of one image: instances with quality q > lam, pasted in decreasing q at threshold t;
    with pmin > 0, instances whose mean refined probability is below pmin are dropped and the rest pasted again."""
    sel2 = [(c, float(qq)) for (c, _), qq in zip(sel, q)]
    use = q > lam
    if pmin > 0:
        pm = np.array([p for _, p in refine_seq(sel2, preds, Ms, t, use=use)])
        use &= pm >= pmin
    res = refine_seq(sel2, preds, Ms, t, use=use)
    return [(r['counts'].decode(), float(qq)) for (r, _), qq in zip(res, q) if r is not None]
