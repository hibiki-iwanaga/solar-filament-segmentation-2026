# %% [markdown]
# # Stage 14 (final run): candidate pool, descriptors, first quality model, greedy selection (CPU)
# Own pipeline. Same code as `filament-v43b-evidence-native` (which runs the code of `filament-v43-physics-descriptors`,
# itself the code of `filament-v38-detector-candidates` with a line patch; all three are attached and read from their
# `__notebook__.ipynb`), with these additions for inference without retraining:
# * the candidate table is built in a fixed image order (the worker pool returns images in completion order, and the
#   bagging of LightGBM depends on the row order), so the fitted models are the same from run to run;
# * the fitted LightGBM models of the selected quality model are saved (`stage1_models/`) and checked against the
#   test predictions;
# * the out-of-fold histograms used to calibrate the test maps are saved (`calibration.json`);
# * the fully patched source is written to `s14_source.py`.
#
# This file is the fully patched source that this run executed (its `s14_source.py`).

# %%
import os, glob, json, time, pickle
from collections import defaultdict
from itertools import combinations
import numpy as np, pandas as pd, cv2
from pycocotools import mask as mu
from multiprocessing import Pool
cv2.setNumThreads(0)   # no OpenCV worker threads in the parent: forked workers would deadlock on them

ROOT = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026', recursive=True)[0]
TRJ = glob.glob(ROOT + '/train/*.json')[0]
TRD = ROOT + '/train/train_images'; TED = ROOT + '/test/test_images'
D1K = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/oof_f0.npz', recursive=True))
def pick(key):
    c = [d for d in D1K if key in d]
    return c[0] if c else None
DIRS = {'B': pick('v5b'), 'C': pick('v12'), 'D': pick('v14')}
DIRS = {k: v for k, v in DIRS.items() if v}
DE = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/oofall_f0.npz', recursive=True))
if DE:
    DIRS['E'] = DE[0]
PAT = {'E': 'oofall_f'}; TPAT = {'E': 'testall_mean.npz'}
DSP = DIRS.get('D')
DET = {}
for _p in sorted(glob.glob('/kaggle/input/**/detections3*.pkl', recursive=True)):
    DET.update(pickle.load(open(_p, 'rb')))
DET.pop('m27', None)
CAND_TAGS = []                                            # v38: detector instances as candidates do not help
SMIN = 0.3
print(ROOT, DIRS, 'detectors', sorted(DET), 'candidates from', CAND_TAGS, flush=True)
J = json.load(open(TRJ))
imgs = {im['id']: im for im in J['images']}
sets = defaultdict(list)
for a in J['annotations']:
    sets[a['image_id']].append(a)
stem_sets = defaultdict(list)
for sid, im in imgs.items():
    stem_sets[os.path.splitext(im['file_name'])[0]].append(sid)
stems = sorted(stem_sets)
H = W = 2048
def ann_rle(a):
    r = mu.frPyObjects(a['segmentation'], H, W)
    return mu.merge(r) if isinstance(r, list) else r
GT = {s: [(sid, [ann_rle(a) for a in sets[sid]]) for sid in stem_sets[s]] for s in stems}
perm = np.random.RandomState(42).permutation(len(stems))
FOLD = {stems[i]: k % 5 for k, i in enumerate(perm)}
KOF = {s: len(GT[s]) for s in stems}
NGT = {s: sum(len(x) for _, x in GT[s]) for s in stems}
tests = sorted(os.path.splitext(f)[0] for f in os.listdir(TED))
TRP = {s: glob.glob(f'{TRD}/{s}.*')[0] for s in stems}
TEP = {s: glob.glob(f'{TED}/{s}.*')[0] for s in tests}

def load_oof(d, pat='oof_f'):
    out = {}
    for f in range(5):
        arr = np.load(f'{d}/{pat}{f}.npz')['a']
        for s, p in zip([s for s in stems if FOLD[s] == f], arr):
            out[s] = p
    return out
OOF = {k: load_oof(d, PAT.get(k, 'oof_f')) for k, d in DIRS.items()}
SP = load_oof(DSP, 'oofsp_f') if DSP else None
MEM = sorted(DIRS)
print('members', MEM, 'spine', SP is not None, flush=True)

def ens(stack):
    """uint8 mean of a list of uint8 maps (rounded)."""
    acc = np.zeros(stack[0].shape, np.uint16)
    for a in stack:
        acc += a
    return ((acc + len(stack) // 2) // len(stack)).astype(np.uint8)

USE2K = True   # v15b: member C (trained on full-resolution crops) at its native 2048 px inference
OOF2 = load_oof(DIRS['C'], 'oof2k_f') if USE2K and 'C' in DIRS else None
def up(a):
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
def ensm(name, s):
    """Ensemble map of one training image: 1024 px, or 2048 px when the 2048 px map of C is used."""
    if OOF2 is not None and 'C' in name:
        return ens([OOF2[s] if k == 'C' else up(OOF[k][s]) for k in name])
    return ens([OOF[k][s] for k in name])
print('USE2K', USE2K, OOF2 is not None, flush=True)
KER = {c: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)) for c in (11, 25)}
RANK = {}
NAMES = ['BCD']
print('full selection on', NAMES, flush=True)
def qmatch(h_oof, te):
    h_te = np.bincount(te.ravel(), minlength=256).astype(float)
    c_oof = np.cumsum(h_oof) / h_oof.sum(); c_te = np.cumsum(h_te) / h_te.sum()
    return np.clip(np.searchsorted(c_oof, c_te, side='left'), 0, 255).astype(np.uint8)

def spread(stack):
    """Mean absolute deviation from the member mean (uint8)."""
    a = np.stack(stack).astype(np.float32)
    return np.abs(a - a.mean(0)).mean(0).round().astype(np.uint8)

TEST, TSTD = {}, {}
for name in NAMES:
    h = np.zeros(256)
    for s in stems:
        h += np.bincount(ensm(name, s).ravel(), minlength=256)
    if OOF2 is not None and 'C' in name:
        tl = {k: np.load(f'{DIRS[k]}/test2k_mean.npz' if k == 'C' else f'{DIRS[k]}/' + TPAT.get(k, 'test_mean.npz'))['a'] for k in name}
        tm = np.stack([ens([tl[k][i] if k == 'C' else up(tl[k][i]) for k in name]) for i in range(len(tests))]); del tl
    else:
        tm = ens([np.load(f'{DIRS[k]}/test_mean.npz')['a'] for k in name])
    TEST[name] = qmatch(h, tm)[tm]; del tm
    CALIB = {'h_' + name: h.tolist()}
    acc = np.zeros((len(tests), 1024, 1024), np.float32)
    for f in range(5):
        acc += spread([np.load(f'{DIRS[k]}/test_f{f}.npz')['a'] for k in name if k != 'E'])
    TSTD[name] = (acc / 5).round().astype(np.uint8); del acc
    print(name, 'mean spread oof / test:', float(np.mean([spread([OOF[k][s] for k in name if k != 'E']).mean() for s in stems[:80]])),
          float(TSTD[name].mean()), flush=True)
if SP is not None:
    hs = np.zeros(256)
    for s in stems:
        hs += np.bincount(SP[s].ravel(), minlength=256)
    tsp = np.load(f'{DSP}/testsp_mean.npz')['a']
    TSP = qmatch(hs, tsp)[tsp]; del tsp
    CALIB['h_spine'] = hs.tolist()
else:
    TSP = None

TS = (0.2, 0.3, 0.4, 0.5, 0.6)
TS_FINE = (0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.8)      # extra levels: 13 thresholds in all
SETTINGS = [(t, c, c) for t in TS for c in (0, 11, 25)] + [(t, 11, g) for t in TS for g in (41, 61)]
KERG = {g: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g // 4 + 1, g // 4 + 1)) for g in (41, 61)}

def _axis(xs, ys):
    """orientation (deg, image y up, -90..90, positive = right end up) and elongation of a pixel set."""
    xm, ym = xs.mean(), ys.mean()
    sxx, syy, sxy = ((xs - xm) ** 2).mean(), ((ys - ym) ** 2).mean(), ((xs - xm) * (ys - ym)).mean()
    th = 0.5 * np.degrees(np.arctan2(-2 * sxy, sxx - syy))
    tr_, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(max(tr_ * tr_ / 4 - det, 0.))
    return float(th), float(np.sqrt(max(tr_ / 2 + disc, 1e-3) / max(tr_ / 2 - disc, 1e-3))), xm, ym
def _adiff(a, b):
    return float(abs((a - b + 90) % 180 - 90))
PHYS = ['ph_lat', 'ph_abslat', 'ph_lon', 'ph_abslon', 'ph_mu', 'ph_r', 'ph_tilt', 'ph_tilt_h', 'ph_abstilt', 'ph_rad_ang', 'ph_llen', 'ph_larea',
        'lk_n', 'lk_gap_max', 'lk_gap_mean', 'lk_gap_rel', 'lk_gap_dep', 'lk_second', 'lk_align', 'lk_tiltdiff', 'lk_elong1', 'lk_fit', 'lk_fit_rel',
        'dark_frac', 'ring_bright', 'ring_p5', 'img_sharp', 'img_std', 'img_R', 'img_limb', 'img_cx', 'img_cy']
def physics(sub, xs, ys, cx, cy, major, a, dcx, dcy, dR):
    """Solar-physics descriptors of one candidate. xs, ys: pixel coordinates inside the box; cx, cy: centroid in the image."""
    ux, uy = (cx - dcx) / dR, (dcy - cy) / dR
    r = float(np.hypot(ux, uy)); mu_ = float(np.sqrt(max(1. - r * r, 0.02)))
    lat = float(np.degrees(np.arcsin(np.clip(uy, -1, 1))))
    lon = float(np.degrees(np.arcsin(np.clip(ux / max(np.cos(np.radians(lat)), 0.05), -1, 1))))
    th, el, xm, ym = _axis(xs, ys)
    rad = _adiff(th, np.degrees(np.arctan2(uy, ux))) if r > 0.02 else 45.
    cr = np.cos(np.radians(rad)); sr = np.sin(np.radians(rad))
    f = dict(ph_lat=lat, ph_abslat=abs(lat), ph_lon=lon, ph_abslon=abs(lon), ph_mu=mu_, ph_r=r, ph_tilt=th, ph_tilt_h=th * (1. if lat >= 0 else -1.),
             ph_abstilt=abs(th), ph_rad_ang=rad, ph_llen=float(np.log(max(major, 1.) * np.sqrt(cr * cr / mu_ ** 2 + sr * sr))), ph_larea=float(np.log(a / mu_)))
    lk = dict(lk_n=1, lk_gap_max=0., lk_gap_mean=0., lk_gap_rel=0., lk_gap_dep=0., lk_second=0., lk_align=0., lk_tiltdiff=0., lk_elong1=el, lk_fit=0., lk_fit_rel=0.)
    # residual of a quadratic spine fit in the principal-axis frame (how well one smooth curve explains all pixels)
    t = np.radians(th); px = (xs - xm) * np.cos(t) - (ys - ym) * np.sin(t); py = (xs - xm) * np.sin(t) + (ys - ym) * np.cos(t)
    step = max(len(px) // 1500, 1)
    try:
        co = np.polyfit(px[::step], py[::step], 2)
        res = float(np.sqrt(np.mean((py[::step] - np.polyval(co, px[::step])) ** 2)))
    except Exception:
        res = 0.
    lk['lk_fit'] = res; lk['lk_fit_rel'] = float(res / max(a / max(major, 1.), 1.))
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 20]
    if len(ids) >= 2:
        ids = sorted(ids, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:8]
        D = np.zeros((len(ids), len(ids)))
        for p_, i in enumerate(ids):
            dt = cv2.distanceTransform((lab != i).astype(np.uint8), cv2.DIST_L2, 3)
            for q_, j in enumerate(ids):
                if p_ != q_:
                    D[p_, q_] = dt[lab == j].min()
        D = np.minimum(D, D.T)
        inn = [0]; edges = []
        while len(inn) < len(ids):
            out_ = [k for k in range(len(ids)) if k not in inn]
            sd_ = D[np.ix_(inn, out_)]
            p_, q_ = np.unravel_index(sd_.argmin(), sd_.shape)
            edges.append(sd_[p_, q_]); inn.append(out_[q_])
        y1, x1 = np.nonzero(lab == ids[0]); y2, x2 = np.nonzero(lab == ids[1])
        t1, e1, xm1, ym1 = _axis(x1, y1); t2, e2, xm2, ym2 = _axis(x2, y2)
        link = np.degrees(np.arctan2(-(ym2 - ym1), xm2 - xm1))
        gmax = float(max(edges))
        lk.update(lk_n=len(ids), lk_gap_max=gmax, lk_gap_mean=float(np.mean(edges)), lk_gap_rel=float(gmax / max(major, 1.)),
                  lk_gap_dep=float(gmax / mu_), lk_second=float(st[ids[1], cv2.CC_STAT_AREA] / a), lk_align=_adiff(t1, link),
                  lk_tiltdiff=_adiff(t1, t2) if e2 > 1.5 else 45., lk_elong1=e1)
    f.update(lk)
    return f

def candidates(prob, std, spine, img_path, extra=()):
    """All candidate instances of one image: list of (rle, feature dict). Maps at 1024 px (uint8).
    extra: detector instances [(rle, score, source index)], described exactly like the pool candidates."""
    cv2.setNumThreads(1)
    p = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    sd = cv2.resize(std, (W, H), interpolation=cv2.INTER_LINEAR)
    sp = cv2.resize(spine, (W, H), interpolation=cv2.INTER_LINEAR) if spine is not None else None
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disk = im[im > 20]; med = float(np.median(disk)) if disk.size else 128.
    dm = cv2.medianBlur(im.astype(np.uint8), 5) > 25                 # solar disk: centre and radius from the image itself
    ysd = np.where(dm.sum(1) > 20)[0]; xsd = np.where(dm.sum(0) > 20)[0]
    if len(ysd) > 100 and len(xsd) > 100:
        dcx, dcy, dR = (xsd[0] + xsd[-1]) / 2, (ysd[0] + ysd[-1]) / 2, ((xsd[-1] - xsd[0]) + (ysd[-1] - ysd[0])) / 4
    else:
        dcx, dcy, dR = W / 2, H / 2, 0.45 * W
    yy, xx = np.ogrid[:H, :W]
    rr2 = ((xx - dcx) ** 2 + (yy - dcy) ** 2) / dR ** 2
    inner = rr2 < 0.64
    lap = cv2.Laplacian(cv2.GaussianBlur(im, (0, 0), 1.), cv2.CV_32F, ksize=3)
    IMG = dict(img_sharp=float(np.log(lap[inner].var() / med ** 2 + 1e-9)), img_std=float(im[inner].std() / med), img_R=float(dR / W),
               img_limb=float(np.median(im[(rr2 > 0.72) & (rr2 < 0.85)]) / max(np.median(im[rr2 < 0.09]), 1.)),
               img_cx=float((dcx - W / 2) / W), img_cy=float((dcy - H / 2) / H))
    del lap, rr2, inner
    out = []
    full = np.zeros((H, W), np.uint8, order='F')
    lo = (p >= 0.1 * 255).astype(np.uint8)
    raw, masks = {}, {}
    def describe(sub, x0, y0, w, h, m0, t, c, g, src, src_score):
        a = int(sub.sum())
        if a < 150:
            return
        pv = p[y0:y0 + h, x0:x0 + w][sub].astype(np.float32) / 255.
        dv = sd[y0:y0 + h, x0:x0 + w][sub].astype(np.float32) / 255.
        iv = im[y0:y0 + h, x0:x0 + w][sub] / med
        pad = 15
        ya, yb, xa, xb = max(0, y0 - pad), min(H, y0 + h + pad), max(0, x0 - pad), min(W, x0 + w + pad)
        big = np.zeros((yb - ya, xb - xa), np.uint8); big[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = sub
        ring = cv2.dilate(big, np.ones((pad, pad), np.uint8)).astype(bool) & ~big.astype(bool)
        rv = im[ya:yb, xa:xb][ring] / med
        pring = p[ya:yb, xa:xb][ring].astype(np.float32) / 255.
        cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(big); cv2.drawContours(filled, cnts, -1, 1, -1)
        fsub = filled[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w].astype(bool) | sub
        fa = int(fsub.sum())
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt); ha = max(cv2.contourArea(hull), 1.)
        peri = cv2.arcLength(cnt, True)
        ys, xs = np.nonzero(sub)
        cx, cy = xs.mean() + x0, ys.mean() + y0
        if len(xs) > 2:
            xm, ym = xs.mean(), ys.mean()
            sxx, syy, sxy = ((xs - xm) ** 2).mean(), ((ys - ym) ** 2).mean(), ((xs - xm) * (ys - ym)).mean()
            tr_, det = sxx + syy, sxx * syy - sxy ** 2
            disc = np.sqrt(max(tr_ * tr_ / 4 - det, 0.))
            elong = float(np.sqrt(max(tr_ / 2 + disc, 1e-3) / max(tr_ / 2 - disc, 1e-3)))
            major = float(np.sqrt(max(tr_ / 2 + disc, 1e-3)) * 4)
        else:
            elong, major = 1., 1.
        core = (m0[y0:y0 + h, x0:x0 + w].astype(bool) & sub)
        ph = physics(sub, xs, ys, cx, cy, major, a, dcx, dcy, dR)
        ph.update(dark_frac=float((iv < 0.55).mean()), ring_bright=float((rv > 1.12).mean()) if rv.size else 0.,
                  ring_p5=float(np.percentile(rv, 5)) if rv.size else 1.)
        nf, _, stf, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
        fr = stf[1:, cv2.CC_STAT_AREA]; fr = fr[fr >= 20]
        f = dict(
            src=src, src_score=src_score, t=t, close=c, group=g, fine=int(t in TS_FINE), larea=np.log(fa), w=w, h=h, pmax=pv.max(), pmean=pv.mean(), pmed=np.median(pv),
            p10=np.percentile(pv, 10), p90=np.percentile(pv, 90), frac5=(pv > .5).mean(), frac7=(pv > .7).mean(),
            pring=pring.mean() if pring.size else 0., smean=dv.mean(), s90=np.percentile(dv, 90),
            imean=iv.mean(), ring=rv.mean() if rv.size else 1., contrast=(rv.mean() - iv.mean()) if rv.size else 0.,
            imin=np.percentile(iv, 5), solidity=a / ha, compact=4 * np.pi * a / max(peri, 1.) ** 2, elong=elong,
            width=a / max(major, 1.), rdisk=np.hypot(cx - W / 2, cy - H / 2) / (W / 2), hole_frac=(fa - a) / fa,
            frag_n=int(len(fr)), frag_max=float(fr.max() / a) if len(fr) else 0., core=float(core.sum() / a))
        f.update(ph)
        if sp is not None:
            sv = sp[y0:y0 + h, x0:x0 + w]
            svin = sv[sub].astype(np.float32) / 255.
            spm = (sv >= 128) & sub
            ns, _, sts, _ = cv2.connectedComponentsWithStats(spm.astype(np.uint8), connectivity=8)
            sl = sts[1:, cv2.CC_STAT_AREA]; sl = sl[sl >= 30]
            f.update(spmax=svin.max(), spmean=svin.mean(), sp_len=float(spm.sum()) / max(major, 1.),
                     sp_n=int(len(sl)), sp_main=float(sl.max() / max(sl.sum(), 1)) if len(sl) else 0.,
                     sp_ring=float((sp[ya:yb, xa:xb][ring] >= 128).mean()) if ring.any() else 0.)
        full[y0:y0 + h, x0:x0 + w] = fsub
        enc = mu.encode(full)
        full[y0:y0 + h, x0:x0 + w] = 0
        out.append((enc, f))
    for t, c, g in SETTINGS:
        if t not in raw:
            raw[t] = (p >= t * 255).astype(np.uint8)
        m0 = raw[t]
        if (t, c) not in masks:
            masks[(t, c)] = (cv2.morphologyEx(m0, cv2.MORPH_CLOSE, KER[c]) & lo | m0) if c else m0
        m = masks[(t, c)]
        grouped = g != c
        if grouped:
            small = m.reshape(H // 4, 4, W // 4, 4).max((1, 3))
            n, gl, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(small, cv2.MORPH_CLOSE, KERG[g]), connectivity=8)
            lab = np.repeat(np.repeat(gl, 4, 0), 4, 1)
            st = st * np.array([4, 4, 4, 4, 16])
        else:
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 150:
                continue
            x0, y0, w, h = st[i, :4]
            sub = (lab[y0:y0 + h, x0:x0 + w] == i)
            if grouped:
                sub &= m[y0:y0 + h, x0:x0 + w].astype(bool)
                k_pieces = cv2.connectedComponents(sub.astype(np.uint8), connectivity=8)[0] - 1
                if k_pieces < 2:
                    continue
            describe(sub, x0, y0, w, h, m0, t, c, g, 0, 0.)
    if 0.5 not in raw:
        raw[0.5] = (p >= 0.5 * 255).astype(np.uint8)
    for r, sc, si in extra:                                   # detector instances: same descriptors, flag src = 1 + source index
        x0, y0, w, h = [int(v) for v in mu.toBbox(r)]
        if w < 1 or h < 1:
            continue
        sub = mu.decode(r)[y0:y0 + h, x0:x0 + w].astype(bool)
        describe(sub, x0, y0, w, h, raw[0.5], 0.5, 0, 0, 1 + si, float(sc))
    n, lab, st, cen = cv2.connectedComponentsWithStats((p >= 0.5 * 255).astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    bigc = areas[areas >= 600]
    for _, f in out:
        f['img_n'] = int(len(bigc)); f['img_larea'] = float(np.log1p(bigc.sum()))
        f['rel_area'] = float(np.exp(f['larea']) / max(bigc.max(), 1)) if len(bigc) else 0.
        f['img_spread'] = float(sd.mean())
    return out

def score(rles, s):
    K = len(GT[s]); Sm = np.zeros((len(rles), K))
    for j, (sid, g) in enumerate(GT[s]):
        if not g or not rles:
            continue
        iou = np.array(mu.iou(rles, g, [0] * len(g)))
        Sm[:, j] = np.where(iou > .5, iou, 0).max(1)
    return Sm, K

def overlaps(cs):
    rles = [c[0] for c in cs]
    if not rles:
        return np.zeros((0, 0), bool)
    ov = np.array(mu.iou(rles, rles, [1] * len(rles)))
    areas = np.array([np.exp(c[1]['larea']) for c in cs])
    for i, c in enumerate(cs):
        inside = ov[:, i] > .9; cont = ov[i, :] > .9
        c[1]['n_inside'] = int(inside.sum() - 1); c[1]['n_cont'] = int(cont.sum() - 1)
        c[1]['area_ratio_up'] = float(areas[i] / areas[cont].max()) if cont.any() else 1.
    return ov > 0

def tfold(s):
    """Fold model whose detections are the candidates of a test image (one model per image, as out of fold)."""
    return sorted(DET['m31']['test'])[tests.index(s) % len(DET['m31']['test'])]
def det_extra(split, s):
    ex = []
    for si, tag in enumerate(CAND_TAGS):
        views = DET[tag]['oof'].get(s) if split == 'train' else DET[tag]['test'][tfold(s)].get(s)
        if views:
            ex += [(dict(size=[H, W], counts=c.encode()), sc, si) for c, sc in views[0] if sc >= SMIN]
    return ex
def work(job):
    name, split, s = job
    if split == 'train':
        cs = candidates(ensm(name, s), spread([OOF[k][s] for k in name if k != 'E']),
                        SP[s] if SP is not None else None, TRP[s], det_extra(split, s))
        Sm, K = score([c[0] for c in cs], s)
    else:
        i = tests.index(s)
        cs = candidates(TEST[name][i], TSTD[name][i], TSP[i] if TSP is not None else None, TEP[s], det_extra(split, s))
        Sm, K = None, None
    ov = overlaps(cs)
    return name, split, s, cs, ov, Sm, K

JOBS = [(n, 'train', s) for n in NAMES for s in stems] + [(n, 'test', s) for n in NAMES for s in tests]
t0 = time.time()
RES = defaultdict(dict)
with Pool(4) as pool:
    for k, r in enumerate(pool.imap_unordered(work, JOBS, chunksize=2)):
        RES[(r[0], r[1])][r[2]] = r[3:]
        if k % 200 == 0:
            print(k, len(JOBS), f'{time.time() - t0:.0f}s', flush=True)
print('candidates done', f'{time.time() - t0:.0f}s', {k: sum(len(v[0]) for v in d.values()) for k, d in RES.items()},
      'from the detector', {k: sum(c[1]['src'] > 0 for v in d.values() for c in v[0]) for k, d in RES.items()}, flush=True)
LD = DIRS.get('E')
SIMF = []
if LD:
    AINFO = pickle.load(open(f'{LD}/annotators21.pkl', 'rb'))
    TEAMI = np.array(AINFO['team']) - 1
    CALIB['h_sim'] = sum(np.load(f'{LD}/hist_f{f}.npy') for f in range(5)).tolist()
    CALIB['sim_annotators'] = len(AINFO['anns']); CALIB['sim_team'] = [int(a) for a in AINFO['team']]
    os.makedirs('/tmp/lev', exist_ok=True)
    LEVG = None
    def dump(a):
        split, i, s = a
        np.savez_compressed(f'/tmp/lev/{split}_{s}.npz', a=LEVG[i])
    t1 = time.time()
    for f in range(5):
        LEVG = np.load(f'{LD}/ooflev_f{f}.npz')['a']
        ss = [s for s in stems if FOLD[s] == f]
        assert len(ss) == len(LEVG), (len(ss), LEVG.shape)
        with Pool(4) as pool:
            pool.map(dump, [('train', i, s) for i, s in enumerate(ss)], chunksize=4)
    LEVG = np.load(f'{LD}/testlev.npz')['a']
    assert len(LEVG) == len(tests)
    with Pool(4) as pool:
        pool.map(dump, [('test', i, s) for i, s in enumerate(tests)], chunksize=4)
    NA = LEVG.shape[1]; del LEVG
    print('simulated annotators', NA, 'team', len(TEAMI), 'maps split per image', f'{time.time() - t1:.0f}s', flush=True)

    def simfeat(job):
        """Descriptors of every candidate of one image against the simulated annotators (maps at 1024 px)."""
        split, s = job
        cv2.setNumThreads(1)
        lev = np.load(f'/tmp/lev/{split}_{s}.npz')['a']
        LAB = {}
        for L in (1, 2, 3):
            for a in range(NA):
                n, lab, st, _ = cv2.connectedComponentsWithStats((lev[a] >= L).astype(np.uint8), connectivity=8)
                LAB[(L, a)] = (lab, st[:, cv2.CC_STAT_AREA].astype(np.float64))
        nper = np.array([(LAB[(2, a)][1][1:] >= 40).sum() for a in range(NA)], float)      # >= 160 px at 2048
        out = {}
        for name in NAMES:
            rows = []
            for r, f in RES[(name, split)][s][0]:
                x0, y0, w, h = [int(v) for v in mu.toBbox(r)]
                m = mu.decode(r)[y0:y0 + h, x0:x0 + w]
                ya, xa = y0 // 2, x0 // 2
                hh, ww = (y0 + h + 1) // 2 - ya, (x0 + w + 1) // 2 - xa
                mm = np.zeros((hh * 2, ww * 2), np.uint8); mm[y0 - ya * 2:y0 - ya * 2 + h, x0 - xa * 2:x0 - xa * 2 + w] = m
                ms = mm.reshape(hh, 2, ww, 2).max((1, 3)).astype(bool)
                ac = float(ms.sum())
                d = {}
                for L, tag in ((2, ''), (1, '_lo'), (3, '_hi')):
                    iou = np.zeros(NA); cov = np.zeros(NA); nt = np.zeros(NA); over = np.full(NA, np.nan)
                    for a in range(NA):
                        lab, area = LAB[(L, a)]
                        ids = lab[ya:ya + hh, xa:xa + ww][ms]
                        if not ids.size or not ids.any():
                            continue
                        cnt = np.bincount(ids, minlength=len(area))[1:].astype(np.float64)
                        b = int(cnt.argmax()); inter = cnt[b]
                        iou[a] = inter / (ac + area[b + 1] - inter); cov[a] = cnt.sum() / ac
                        nt[a] = (cnt >= max(5., 0.05 * ac)).sum(); over[a] = (area[b + 1] - inter) / area[b + 1]
                    hit = iou > .5
                    d['sim_q' + tag] = float((iou * hit).mean()); d['sim_m' + tag] = float(hit.mean())
                    if L == 2:
                        d.update(sim_iou=float(iou.mean()), sim_iou_sd=float(iou.std()), sim_cov=float(cov.mean()), sim_any=float((cov > .3).mean()),
                                 sim_nt=float(nt.mean()), sim_over=float(np.nanmean(over)) if np.isfinite(over).any() else 1.,
                                 sim_q_team=float((iou * hit)[TEAMI].mean()), sim_m_team=float(hit[TEAMI].mean()),
                                 sim_q_solo=float(np.delete(iou * hit, TEAMI).mean()) if len(TEAMI) < NA else 0.,
                                 sim_iou_max=float(iou.max()), sim_iou_q25=float(np.percentile(iou, 25)))
                d['sim_qmax'] = max(d['sim_q'], d['sim_q_lo'], d['sim_q_hi'])
                d['sim_n_img'] = float(nper.mean()); d['sim_n_sd'] = float(nper.std())
                rows.append(d)
            out[name] = rows
        return split, s, out
    t1 = time.time()
    with Pool(4) as pool:
        for k, (split, s, out) in enumerate(pool.imap_unordered(simfeat, [('train', s) for s in stems] + [('test', s) for s in tests], chunksize=2)):
            for name, rows in out.items():
                for (r, f), d in zip(RES[(name, split)][s][0], rows):
                    f.update(d)
            if k % 200 == 0:
                print('sim', k, f'{time.time() - t1:.0f}s', flush=True)
    SIMF = sorted({k for s in stems[:50] for c in RES[(NAMES[0], 'train')][s][0][:1] for k in c[1] if k.startswith('sim_')})
    print('simulated-annotator descriptors', SIMF, f'{time.time() - t1:.0f}s', flush=True)
DETF = ['det_iou', 'det_score', 'det_q', 'det_iou5', 'det_cov', 'det_n', 'det_smax', 'det_in', 'det_nimg']
def det_desc(rles, dets):
    """Descriptors of the candidates of one image against one list of detector instances [(counts, score)]."""
    n = len(rles)
    out = {k: np.zeros(n) for k in DETF}
    if not n or not dets:
        return out
    dr = [dict(size=[H, W], counts=c.encode()) for c, _ in dets]; sc = np.array([s for _, s in dets])
    iou = np.array(mu.iou(rles, dr, [0] * len(dr))).reshape(n, len(dr))
    cov = np.array(mu.iou(rles, dr, [1] * len(dr))).reshape(n, len(dr))        # intersection / candidate area
    j = iou.argmax(1); best = iou.max(1)
    out['det_iou'] = best; out['det_score'] = sc[j] * (best > 0)
    out['det_q'] = (iou * (iou > .5) * sc[None]).max(1)
    hi = sc >= .5
    if hi.any():
        out['det_iou5'] = iou[:, hi].max(1)
    mid = sc >= .3
    if mid.any():
        U = mu.merge([d for d, m in zip(dr, mid) if m])
        out['det_cov'] = np.array(mu.iou(rles, [U], [1])).reshape(n)
    out['det_n'] = ((cov >= .1) & mid[None]).sum(1).astype(float)
    out['det_smax'] = np.where(cov >= .5, sc[None], 0.).max(1)
    cj = cov[np.arange(n), j]                                                        # share of the best instance inside the candidate:
    with np.errstate(divide='ignore', invalid='ignore'):                             # I / A_det = 1 / (1 / IoU - 1 / coverage + 1), no areas needed
        out['det_in'] = np.where(best > 0, 1. / (1. / np.maximum(best, 1e-9) - 1. / np.maximum(cj, 1e-9) + 1.), 0.)
    out['det_nimg'] = np.full(n, float(hi.sum()))
    return out
TAGS = [t for t in ('m31', 'm34') if t in DET]
NV = len(next(iter(DET[TAGS[0]]['oof'].values())))
GROUPS = {'d2': [('m31', v) for v in range(NV)]}
if 'm34' in DET:
    GROUPS['d3'] = [('m34', v) for v in range(NV)]
    GROUPS['d23'] = GROUPS['d2'] + GROUPS['d3']
def detfeat(job):
    name, split, s = job
    rles = [c[0] for c in RESC[(name, split)][s][0]]
    src = np.array([c[1]['src'] for c in RESC[(name, split)][s][0]])
    per = {}
    for tag in TAGS:
        for v in range(NV):
            if split == 'train':
                ds = [det_desc(rles, DET[tag]['oof'].get(s, [[]] * NV)[v])]
                per[(tag, v)] = ds[0]
            else:
                fs = sorted(DET[tag]['test'])
                ds = [det_desc(rles, DET[tag]['test'][f].get(s, [[]] * NV)[v]) for f in fs]
                own = ds[0]              # a detector candidate is described by its own model only (as out of fold)
                mine = src == 1 + CAND_TAGS.index(tag) if tag in CAND_TAGS else np.zeros(len(rles), bool)
                per[(tag, v)] = {k: np.where(mine, own[k], np.mean([x[k] for x in ds], 0)) for k in DETF}
    d = {}
    for g, members in GROUPS.items():
        for k in DETF:
            d[k.replace('det_', g + '_')] = np.mean([per[m][k] for m in members], 0)
        if len(members) > 1:
            d[g + '_iou_sd'] = np.std([per[m]['det_iou'] for m in members], 0)      # disagreement of the views / detectors
            d[g + '_q_min'] = np.min([per[m]['det_q'] for m in members], 0)
    Sm, K = score(rles, s) if split == 'train' else (None, None)
    return name, split, s, d, Sm, K
RESC = {k: {s: (v[0], v[1]) for s, v in d.items()} for k, d in RES.items()}
RES = defaultdict(dict)
JOBS = [(n, 'train', s) for n in NAMES for s in RESC[(n, 'train')]] + [(n, 'test', s) for n in NAMES for s in RESC[(n, 'test')]]
t0 = time.time()
with Pool(4) as pool:
    for k, (name, split, s, d, Sm, K) in enumerate(pool.imap_unordered(detfeat, JOBS, chunksize=2)):
        cs, ov = RESC[(name, split)][s][:2]
        for i, (r, f) in enumerate(cs):
            for key, arr in d.items():
                f[key] = float(arr[i])
        RES[(name, split)][s] = (cs, ov, Sm, K)
        if k % 200 == 0:
            print('det', k, len(JOBS), f'{time.time() - t0:.0f}s', flush=True)
print('detector descriptors done', sorted(GROUPS), f'{time.time() - t0:.0f}s', flush=True)
RES = {k: {s: d[s] for s in (stems if k[1] == 'train' else tests) if s in d} for k, d in RES.items()}   # fixed image order
pickle.dump({k: {s: ([(c[0], {a: b for a, b in c[1].items() if a not in PHYS}) for c in v[0]], v[1]) for s, v in d.items()} for k, d in RES.items()}, open('/kaggle/working/candidates_v43.pkl', 'wb'))
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
name = NAMES[0]
rows = []
for s, (cs, ov, Sm, K) in RES[(name, 'train')].items():
    for k, (r, f) in enumerate(cs):
        d = dict(f); d.update(stem=s, idx=k, q=Sm[k].mean(), K=K, S=Sm[k].sum(), hit=float((Sm[k] > 0).mean())); rows.append(d)
D = pd.DataFrame(rows); D['fold'] = D.stem.map(FOLD)
def greedy(pred, ov, lam, allowed=None):
    order = np.argsort(-pred); taken = np.zeros(len(pred), bool); blocked = np.zeros(len(pred), bool)
    for i in order:
        if pred[i] <= lam:
            break
        if blocked[i] or (allowed is not None and not allowed[i]):
            continue
        taken[i] = True; blocked |= ov[i]
    return taken
SUBS = {'all': stems, 'K1': [s for s in stems if KOF[s] == 1], 'K2+': [s for s in stems if KOF[s] >= 2]}
G = {s: g for s, g in D.groupby('stem')}
for s, g in G.items():
    assert (g.idx.values == np.arange(len(g))).all()
OV = {s: v[1] for s, v in RES[(name, 'train')].items()}
def pq(TK, subset):
    num = den = 0.
    for s in subset:
        den += .5 * NGT[s]
        if s not in G:
            continue
        g = G[s]; tk = TK[s]
        num += g.S.values[tk].sum(); den += .5 * g.K.values[0] * tk.sum()
    return num / den
PRM = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4)
DROP = ('stem', 'idx', 'q', 'S', 'K', 'fold', 'hit', 'fine')
PREF = tuple(g + '_' for g in GROUPS)
FE0 = [c for c in D.columns if c not in DROP and not c.startswith(PREF)]
FSETS = {}
for g in GROUPS:
    FSETS[g] = FE0 + [c for c in D.columns if c.startswith(g + '_')]
if 'd3' in GROUPS:
    FSETS['dcat'] = FE0 + [c for c in D.columns if c.startswith(('d2_', 'd3_'))]
BASE = 'd3' if 'd3' in GROUPS else 'd2'
FSETS[BASE + 'np'] = [c for c in FSETS[BASE] if c not in PHYS]              # same model without the physics descriptors
FSETS[BASE + 'pos'] = [c for c in FSETS[BASE] if not c.startswith('lk_')]      # physics without the linkage geometry
print('physics descriptors', [c for c in PHYS if c in D.columns], 'missing', [c for c in PHYS if c not in D.columns], flush=True)
print(D[[c for c in PHYS if c in D.columns]].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string(), flush=True)
print('correlation with q:', {c: round(float(np.corrcoef(D[c], D.q)[0, 1]), 3) for c in PHYS if c in D.columns and D[c].std() > 0}, flush=True)
FSETS = {k: [c for c in v if c not in PHYS] for k, v in FSETS.items() if not k.endswith(('np', 'pos'))}
FSETS[BASE + 'ph'] = FSETS[BASE] + [c for c in D.columns if c.startswith('ph_')]
POOLONLY = {s: g.src.values == 0 for s, g in D.groupby('stem')}
print('candidates per image: pool', round(float(np.mean([v.sum() for v in POOLONLY.values()])), 1), 'detector', round(float(np.mean([(~v).sum() for v in POOLONLY.values()])), 1),
      '| mean q pool', round(float(D.q[D.src == 0].mean()), 4), 'detector', round(float(D.q[D.src > 0].mean()), 4), flush=True)
print({k: len(v) for k, v in FSETS.items()}, 'candidates', len(D), flush=True)
def fit_hurdle(Xtr, D_, Xte, keep=None):
    """Hurdle model: P(some annotator matches) x expected matched IoU given a match. keep: list that receives the two models."""
    m_hit = lgb.LGBMRegressor(**PRM).fit(Xtr, D_.hit.values)
    ph = m_hit.predict(Xte)
    pos = D_.hit.values > 0
    m_iou = lgb.LGBMRegressor(**PRM).fit(Xtr[pos], (D_.q.values / np.maximum(D_.hit.values, 1e-9))[pos])
    pi = m_iou.predict(Xte)
    if keep is not None:
        keep += [m_hit, m_iou]
    return np.clip(ph, 0, 1) * np.clip(pi, 0, 1)
LAMS = [round(x, 2) for x in np.arange(0.10, 0.44, 0.02)]
res = {}; fo = D.fold.values; t1 = time.time()
for tag, fe in FSETS.items():
    X = D[fe].values.astype(np.float32)
    pr = np.zeros(len(D)); ph = np.zeros(len(D))
    for f in range(5):
        tr, va = fo != f, fo == f
        pr[va] = lgb.LGBMRegressor(**PRM).fit(X[tr], D.q.values[tr]).predict(X[va])
        ph[va] = fit_hurdle(X[tr], D[tr], X[va])
    D['pred_' + tag] = pr; D['pred_' + tag + '_avg'] = 0.5 * pr + 0.5 * ph
    for col in ('pred_' + tag, 'pred_' + tag + '_avg'):
        t = col[5:]
        for lam in LAMS:
            TK = {s: greedy(g[col].values, OV[s], lam) for s, g in D.groupby('stem')}
            r = {k: pq(TK, v) for k, v in SUBS.items()}; r['per_img'] = float(np.mean([v.sum() for v in TK.values()]))
            r['det_img'] = float(np.mean([(v & ~POOLONLY[s]).sum() for s, v in TK.items()]))
            r['pool_only'] = r['all']
            res[(t, lam)] = r
        b = max(LAMS, key=lambda l: res[(t, l)]['all'])
        print(f'{t:9s} same model, selection restricted to the pool candidates:', ' '.join(f"{l}:{res[(t, l)]['pool_only']:.4f}" for l in LAMS[4:12]), flush=True)
        print(f'{t:9s} best lam {b:.2f}', {k: round(v, 4) for k, v in res[(t, b)].items()}, '|', ' '.join(f"{l}:{res[(t, l)]['all']:.4f}" for l in LAMS[4:12]),
              '| corr with q', round(float(np.corrcoef(D[col], D.q)[0, 1]), 4), f'{time.time() - t1:.0f}s', flush=True)
json.dump({'_'.join(map(str, k)): v for k, v in res.items()}, open('/kaggle/working/summary_v43.json', 'w'), indent=1, default=float)
_m = lgb.LGBMRegressor(importance_type='gain', **PRM).fit(D[FSETS[BASE]].values.astype(np.float32), D.q.values)
_imp = sorted(zip(FSETS[BASE], _m.feature_importances_ / _m.feature_importances_.sum()), key=lambda kv: -kv[1])
print('gain share of the physics descriptors', round(float(sum(v for k, v in _imp if k in PHYS)), 4))
print('top 45 by gain (* = physics):', [(('*' if k in PHYS else '') + k, round(float(v), 4)) for k, v in _imp[:45]], flush=True)
print('physics descriptors by gain:', [(k, round(float(v), 4), 'rank %d' % (i + 1)) for i, (k, v) in enumerate(_imp) if k in PHYS], flush=True)
def best_of(t):
    return max(res[(t, l)]['all'] for l in LAMS)
def flat(t):       # mean over the three lambdas around the optimum: less sensitive to one lucky lambda
    b = max(range(len(LAMS)), key=lambda i: res[(t, LAMS[i])]['all'])
    return float(np.mean([res[(t, LAMS[i])]['all'] for i in range(max(b - 1, 0), min(b + 2, len(LAMS)))]))
TG = sorted({k[0] for k in res}, key=lambda t: -flat(t))
print('ranking (best | mean of 3 lambdas around the best):', [(t, round(best_of(t), 4), round(flat(t), 4)) for t in TG], flush=True)
LAM_LOW = 0.10
WIN = TG[0]
KEEP = [WIN]
TS_ = [s for s in tests if RES[(name, 'test')][s][0]]
cut = np.cumsum([0] + [len(RES[(name, 'test')][s][0]) for s in TS_])
SELS = {}
for t in KEEP:
    fs = t[:-4] if t.endswith('_avg') else t
    fe = FSETS[fs]
    X = D[fe].values.astype(np.float32)
    XT = np.concatenate([pd.DataFrame([c[1] for c in RES[(name, 'test')][s][0]])[fe].values.astype(np.float32) for s in TS_])
    m_reg = lgb.LGBMRegressor(**PRM).fit(X, D.q.values)
    pt = m_reg.predict(XT)
    kept_models = []
    if t.endswith('_avg'):
        pt = 0.5 * pt + 0.5 * fit_hurdle(X, D, XT, kept_models)
    os.makedirs('/kaggle/working/stage1_models', exist_ok=True)
    for nm, m in zip(('reg', 'hit', 'iou'), [m_reg] + kept_models):
        m.booster_.save_model(f'/kaggle/working/stage1_models/{nm}.txt')
    _b = {nm: lgb.Booster(model_file=f'/kaggle/working/stage1_models/{nm}.txt') for nm in ('reg', 'hit', 'iou')[:1 + len(kept_models)]}
    _p = _b['reg'].predict(XT)
    if kept_models:
        _p = 0.5 * _p + 0.5 * np.clip(_b['hit'].predict(XT), 0, 1) * np.clip(_b['iou'].predict(XT), 0, 1)
    print('saved boosters reproduce the test predictions: max abs difference', float(np.abs(_p - pt).max()), flush=True)
    json.dump(dict(variant=f'{name}_{t}', features=list(fe), hurdle=bool(kept_models), lam_low=LAM_LOW, lam_best=float(max(LAMS, key=lambda l: res[(t, l)]['all'])),
                   members=name, use2k=USE2K, detector_group=fs, settings=[list(x) for x in SETTINGS], pq_oof=float(max(res[(t, l)]['all'] for l in LAMS))),
              open('/kaggle/working/stage1_models/model.json', 'w'), indent=1)
    TEST_Q = {s: pt[cut[i]:cut[i + 1]] for i, s in enumerate(TS_)}
    col = 'pred_' + t
    best = max(LAMS, key=lambda l: res[(t, l)]['all'])
    sel_tr, sel_te = {}, {}
    for s, g in D.groupby('stem'):
        cs = RES[(name, 'train')][s][0]
        tk = greedy(g[col].values, OV[s], LAM_LOW)
        sel_tr[s] = [(cs[i][0]['counts'].decode(), float(q)) for i, q in zip(np.where(tk)[0], g[col].values[tk])]
    for i, s in enumerate(TS_):
        cs, ov = RES[(name, 'test')][s][:2]
        pr = pt[cut[i]:cut[i + 1]]
        tk = greedy(pr, ov, LAM_LOW)
        sel_te[s] = [(cs[j][0]['counts'].decode(), float(pr[j])) for j in np.where(tk)[0]]
    key = f'{name}_{t}'
    SELS[key] = dict(lam=LAM_LOW, lam_best=best, name=name, train=sel_tr, test=sel_te)
    print(key, 'best lam', best, 'pq all', round(res[(t, best)]['all'], 4), '| kept per image at lam_best: oof',
          round(float(np.mean([sum(q > best for _, q in v) for v in sel_tr.values()])), 2), 'test',
          round(float(np.mean([sum(q > best for _, q in sel_te.get(s, [])) for s in tests])), 2), '| mean test / oof quality of kept',
          round(float(np.mean([q for v in sel_te.values() for _, q in v])), 4), round(float(np.mean([q for v in sel_tr.values() for _, q in v])), 4), flush=True)
pickle.dump(dict(winner=f'{name}_{WIN}', vars=list(SELS), sel=SELS, use2k=USE2K), open('/kaggle/working/selected_v43.pkl', 'wb'))
json.dump(CALIB, open('/kaggle/working/calibration.json', 'w'))
print('saved', sorted(os.listdir('/kaggle/working')), flush=True)

# %% [markdown]
# ## Checks: test-side intermediate results for the inference notebook, comparison with the earlier run

# %%
import hashlib, platform, sklearn, scipy, lightgbm
# test candidates in the order used above, with the predicted quality (the inference notebook must reproduce these)
pickle.dump({s: [(c[0]['counts'].decode() if isinstance(c[0]['counts'], bytes) else c[0]['counts'], float(q)) for c, q in zip(RES[(name, 'test')][s][0], TEST_Q[s])]
             for s in TS_}, open('/kaggle/working/test_candidates_q.pkl', 'wb'))
_prev = [p for p in glob.glob('/kaggle/input/**/selected_v43.pkl', recursive=True) if 'filament-v43b' in p]
if _prev:
    P = pickle.load(open(_prev[0], 'rb'))
    print('earlier run', _prev[0], 'winner', P['winner'], 'now', f'{name}_{WIN}', flush=True)
    for key in SELS:
        if key in P['sel']:
            for sp in ('train', 'test'):
                a, b = P['sel'][key][sp], SELS[key][sp]
                same = sum(1 for s in a if [c for c, _ in a[s]] == [c for c, _ in b.get(s, [])])
                print(key, sp, 'images', len(a), len(b), 'identical instance lists', same, flush=True)
print('md5 selected_v43.pkl', hashlib.md5(open('/kaggle/working/selected_v43.pkl', 'rb').read()).hexdigest())
print('versions', dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, cv2=cv2.__version__, lightgbm=lightgbm.__version__,
                       sklearn=sklearn.__version__, scipy=scipy.__version__), flush=True)
print(sorted(os.listdir('/kaggle/working')), sorted(os.listdir('/kaggle/working/stage1_models')))
