# Training stage 08: source of the Kaggle notebook `filament-v15c-lowlam` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v15c: the v15b selection (B + C at 2048 px + D) saved down to a low keep threshold (CPU)
# Same as v15b, restricted to the winning combination BCD. The greedy disjoint selection visits candidates in
# decreasing predicted quality and stops at the threshold lam, so the selection at a low lam contains the selection
# at any higher lam. The low-lam selection (with the predicted quality of every instance) is saved as seeds for the
# stage-2 refiners, so that the keep threshold can be re-tuned after refinement.
# Own pipeline (v11 selection: candidate pool from thresholds x gap closing x fragment grouping, hole filling,
# LightGBM quality model, greedy pixel-disjoint selection, test map quantile-matched to the out-of-fold values).
#
# Members (5-fold out-of-fold maps at 1024 px, same folds):
#   A  v5A  U-Net ResNet34, full-resolution crops
#   B  v5B  U-Net EfficientNet-B4 at 1024 px
#   C  v12  U-Net ResNet34 with a limb-darkening-flattened second input channel
#   D  v14  UNet++ ResNet34, 2-channel input, soft annotator-mean mask + filament spine head, EMA
# Step 1 ranks member combinations with a cheap fixed recipe (one threshold, gap closing, keep everything, and the
# oracle keep rule); step 2 runs the full selection on the two best combinations with new descriptors:
# spine response inside the candidate (strength, length relative to the candidate size, number of separate spines)
# and the spread between the member networks (on the test set taken per fold, so it means the same as out-of-fold).

# %%
import os, glob, json, time
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
DIRS = {'A': pick('v5a'), 'B': pick('v5b'), 'C': pick('v12'), 'D': pick('v14')}
DIRS = {k: v for k, v in DIRS.items() if v}
DSP = DIRS.get('D')
print(ROOT, DIRS, flush=True)
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
OOF = {k: load_oof(d) for k, d in DIRS.items()}
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

# %% [markdown]
# ## Step 1: rank member combinations with a fixed recipe

# %%
KER = {c: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)) for c in (11, 25)}
RANK = {}
NAMES = ['BCD']
print('full selection on', NAMES, flush=True)

# %% [markdown]
# ## Step 2: full selection on the two best combinations

# %%
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
        tl = {k: np.load(f'{DIRS[k]}/test2k_mean.npz' if k == 'C' else f'{DIRS[k]}/test_mean.npz')['a'] for k in name}
        tm = np.stack([ens([tl[k][i] if k == 'C' else up(tl[k][i]) for k in name]) for i in range(len(tests))]); del tl
    else:
        tm = ens([np.load(f'{DIRS[k]}/test_mean.npz')['a'] for k in name])
    TEST[name] = qmatch(h, tm)[tm]; del tm
    acc = np.zeros((len(tests), 1024, 1024), np.float32)
    for f in range(5):
        acc += spread([np.load(f'{DIRS[k]}/test_f{f}.npz')['a'] for k in name])
    TSTD[name] = (acc / 5).round().astype(np.uint8); del acc
    print(name, 'mean spread oof / test:', float(np.mean([spread([OOF[k][s] for k in name]).mean() for s in stems[:80]])),
          float(TSTD[name].mean()), flush=True)
if SP is not None:
    hs = np.zeros(256)
    for s in stems:
        hs += np.bincount(SP[s].ravel(), minlength=256)
    tsp = np.load(f'{DSP}/testsp_mean.npz')['a']
    TSP = qmatch(hs, tsp)[tsp]; del tsp
else:
    TSP = None

TS = (0.2, 0.3, 0.4, 0.5, 0.6)
SETTINGS = [(t, c, c) for t in TS for c in (0, 11, 25)] + [(t, 11, g) for t in TS for g in (41, 61)]
KERG = {g: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g // 4 + 1, g // 4 + 1)) for g in (41, 61)}

def candidates(prob, std, spine, img_path):
    """All candidate instances of one image: list of (rle, feature dict). Maps at 1024 px (uint8)."""
    cv2.setNumThreads(1)
    p = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    sd = cv2.resize(std, (W, H), interpolation=cv2.INTER_LINEAR)
    sp = cv2.resize(spine, (W, H), interpolation=cv2.INTER_LINEAR) if spine is not None else None
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    disk = im[im > 20]; med = float(np.median(disk)) if disk.size else 128.
    out = []
    full = np.zeros((H, W), np.uint8, order='F')
    lo = (p >= 0.1 * 255).astype(np.uint8)
    raw, masks = {}, {}
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
            a = int(sub.sum())
            if a < 150:
                continue
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
            nf, _, stf, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
            fr = stf[1:, cv2.CC_STAT_AREA]; fr = fr[fr >= 20]
            f = dict(
                t=t, close=c, group=g, larea=np.log(fa), w=w, h=h, pmax=pv.max(), pmean=pv.mean(), pmed=np.median(pv),
                p10=np.percentile(pv, 10), p90=np.percentile(pv, 90), frac5=(pv > .5).mean(), frac7=(pv > .7).mean(),
                pring=pring.mean() if pring.size else 0., smean=dv.mean(), s90=np.percentile(dv, 90),
                imean=iv.mean(), ring=rv.mean() if rv.size else 1., contrast=(rv.mean() - iv.mean()) if rv.size else 0.,
                imin=np.percentile(iv, 5), solidity=a / ha, compact=4 * np.pi * a / max(peri, 1.) ** 2, elong=elong,
                width=a / max(major, 1.), rdisk=np.hypot(cx - W / 2, cy - H / 2) / (W / 2), hole_frac=(fa - a) / fa,
                frag_n=int(len(fr)), frag_max=float(fr.max() / a) if len(fr) else 0., core=float(core.sum() / a))
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

def work(job):
    name, split, s = job
    if split == 'train':
        cs = candidates(ensm(name, s), spread([OOF[k][s] for k in name]),
                        SP[s] if SP is not None else None, TRP[s])
        Sm, K = score([c[0] for c in cs], s)
    else:
        i = tests.index(s)
        cs = candidates(TEST[name][i], TSTD[name][i], TSP[i] if TSP is not None else None, TEP[s])
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
print('candidates done', f'{time.time() - t0:.0f}s', {k: sum(len(v[0]) for v in d.values()) for k, d in RES.items()}, flush=True)

# %%
import lightgbm as lgb   # imported only after the worker pools are closed

def table(name):
    rows = []
    for s, (cs, ov, Sm, K) in RES[(name, 'train')].items():
        for k, (r, f) in enumerate(cs):
            d = dict(f)
            d.update(stem=s, idx=k, q=Sm[k].mean(), K=K, S=Sm[k].sum())
            rows.append(d)
    D = pd.DataFrame(rows)
    D['fold'] = D.stem.map(FOLD)
    return D

SUBS = {'all': stems, 'K1': [s for s in stems if KOF[s] == 1], 'K2+': [s for s in stems if KOF[s] >= 2]}

def greedy(pred, ov, lam):
    order = np.argsort(-pred); taken = np.zeros(len(pred), bool); blocked = np.zeros(len(pred), bool)
    for i in order:
        if pred[i] <= lam:
            break
        if blocked[i]:
            continue
        taken[i] = True; blocked |= ov[i]
    return taken

def pq(TK, G, subset, w=1.):
    num = den = 0.
    for s in subset:
        den += .5 * w * NGT[s]
        if s not in G:
            continue
        g = G[s]; tk = TK[s]
        num += w * g.S.values[tk].sum(); den += .5 * g.K.values[0] * tk.sum()
    return num / den
W_EMPTY = 285 / 439

PRM = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4)
DROP = ('stem', 'idx', 'q', 'S', 'K', 'fold')
res, TABLES, BEST = {}, {}, {}
for name in NAMES:
    D = table(name)
    FEATS = [c for c in D.columns if c not in DROP]
    OV = {s: v[1] for s, v in RES[(name, 'train')].items()}
    for tag, feats in (('new', FEATS), ('v11feats', [c for c in FEATS if not c.startswith(('sp', 's90', 'smean', 'img_spread', 'pring', 'width'))])):
        D['pred'] = 0.
        for f in range(5):
            tr, va = D.fold != f, D.fold == f
            D.loc[va, 'pred'] = lgb.LGBMRegressor(**PRM).fit(D.loc[tr, feats], D.loc[tr, 'q']).predict(D.loc[va, feats])
        D['pred_' + tag] = D['pred']
        G = {s: g for s, g in D.groupby('stem')}
        for lam in np.arange(0.10, 0.44, 0.02):
            TK = {s: greedy(g.pred.values, OV[s][np.ix_(g.idx.values, g.idx.values)], lam) for s, g in G.items()}
            r = {k: pq(TK, G, v) for k, v in SUBS.items()}
            r['K2+empty'] = pq(TK, G, SUBS['K2+'], W_EMPTY)
            r['per_img'] = float(np.mean([v.sum() for v in TK.values()]))
            res[(name, tag, round(float(lam), 2))] = r
            print('   ', name, tag, round(float(lam), 2), {k: round(v, 4) for k, v in r.items()}, flush=True)
    BEST[name] = max((k for k in res if k[0] == name and k[1] == 'new'), key=lambda k: res[k]['all'])
    TABLES[name] = (D, FEATS)
    print('BEST', name, BEST[name], {k: round(v, 4) for k, v in res[BEST[name]].items()}, flush=True)
    imp = pd.Series(lgb.LGBMRegressor(**PRM).fit(D[FEATS], D.q).feature_importances_, FEATS).sort_values(ascending=False)
    print(imp.head(20).to_dict(), flush=True)

# %%
SEL = {}   # selected instances (rle counts, predicted quality) per image, out-of-fold and test: seeds for a stage-2 refiner
LAM_LOW = 0.10
def make(name, lam, path, lam_save=None):
    lam_save = lam if lam_save is None else lam_save
    D, FEATS = TABLES[name]
    model = lgb.LGBMRegressor(**PRM).fit(D[FEATS], D.q)
    rows, sel_te, sel_tr = [], {}, {}
    for s in tests:
        cs, ov = RES[(name, 'test')][s][:2]
        if not cs:
            continue
        pr = model.predict(pd.DataFrame([c[1] for c in cs])[FEATS])
        tk = greedy(pr, ov, lam)
        for k, i in enumerate(np.where(tk)[0]):
            rows.append((f'{s}_{k + 1}', cs[i][0]['counts'].decode()))
        tk = greedy(pr, ov, lam_save)
        sel_te[s] = [(cs[i][0]['counts'].decode(), float(pr[i])) for i in np.where(tk)[0]]
    for s, g in D.groupby('stem'):
        cs, ov = RES[(name, 'train')][s][:2]
        tk = greedy(g.pred_new.values, ov[np.ix_(g.idx.values, g.idx.values)], lam_save)
        sel_tr[s] = [(cs[i][0]['counts'].decode(), float(p_)) for i, p_ in zip(g.idx.values[tk], g.pred_new.values[tk])]
    SEL[name] = dict(lam=lam_save, lam_best=lam, train=sel_tr, test=sel_te)
    sub = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    sub.to_csv(path, index=False)
    print(path, sub.shape, f'{len(sub) / len(tests):.2f} per image', flush=True)

win = max(NAMES, key=lambda n: res[BEST[n]]['all'])
print('winner', win, BEST[win], flush=True)
make(win, BEST[win][2], '/kaggle/working/submission.csv', LAM_LOW)
print('saved selection: lam', LAM_LOW, 'train per image', np.mean([len(v) for v in SEL[win]['train'].values()]),
      'test per image', np.mean([len(v) for v in SEL[win]['test'].values()]), flush=True)
import pickle
pickle.dump(dict(winner=win, sel=SEL, use2k=USE2K), open('/kaggle/working/selected_v15.pkl', 'wb'))
json.dump(dict(rank=RANK, best={n: [list(v), res[v]] for n, v in BEST.items()}, winner=win,
               ladder={f'{k[0]}_{k[1]}_{k[2]}': v for k, v in res.items()}),
          open('/kaggle/working/summary_v15.json', 'w'), indent=1, default=float)
