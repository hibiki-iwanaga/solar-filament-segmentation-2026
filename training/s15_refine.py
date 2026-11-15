# %% [markdown]
# Stage 15: stage-2 refiners applied to the selection of stage 14 (inference only, 2 GPUs).
# Every selected instance is redrawn in an oriented crop by the three saved refiners (stages 09-11); the mean refined probability and the crop geometry are saved for stage 16.
# This file is the fully patched source that `filament-s15-refine-final` executed: the code of `filament-v35-refine-v32` with the line patch of its loader.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.3.4'], check=False)

# %%
import os, glob, json, time, pickle
from collections import defaultdict
import numpy as np, pandas as pd, cv2
from pycocotools import mask as mu
from multiprocessing import Pool
cv2.setNumThreads(0)

ROOT = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026', recursive=True)[0]
TRJ = glob.glob(ROOT + '/train/*.json')[0]
TRD = ROOT + '/train/train_images'; TED = ROOT + '/test/test_images'
SELP = (glob.glob('/kaggle/input/**/selected_v43.pkl', recursive=True) + glob.glob('/kaggle/input/**/selected_v26.pkl', recursive=True))[0]
SD = pickle.load(open(SELP, 'rb'))
VARS = [SD['winner']] + [v for v in SD['vars'] if v != SD['winner']]
SD['winner'] = 'BCD'          # the maps of the crops are always the BCD ensemble (the refiners were trained on them)
SELS = {v: SD['sel'][v] for v in VARS}
D1K = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/oof_f0.npz', recursive=True))
def pick(key):
    c = [d for d in D1K if key in d]
    return c[0] if c else None
DIRS = {'A': pick('v5a'), 'B': pick('v5b'), 'C': pick('v12'), 'D': pick('v14')}
DIRS = {k: v for k, v in DIRS.items() if v}
DSP = DIRS.get('D')
print(ROOT, SELP, {v: (SELS[v]['lam'], SELS[v]['lam_best']) for v in VARS}, DIRS, flush=True)
J = json.load(open(TRJ))
imgs = {im['id']: im for im in J['images']}
sets = defaultdict(list)
for a in J['annotations']:
    sets[a['image_id']].append(a)
stem_sets = defaultdict(list)
for sid, im in imgs.items():
    stem_sets[os.path.splitext(im['file_name'])[0]].append(sid)
stems = sorted(stem_sets)
H = W = 2048; CW, CH = 384, 128
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
def ens(stack):
    acc = np.zeros(stack[0].shape, np.uint16)
    for a in stack:
        acc += a
    return ((acc + len(stack) // 2) // len(stack)).astype(np.uint8)
def qmatch(h_oof, te):
    h_te = np.bincount(te.ravel(), minlength=256).astype(float)
    c_oof = np.cumsum(h_oof) / h_oof.sum(); c_te = np.cumsum(h_te) / h_te.sum()
    return np.clip(np.searchsorted(c_oof, c_te, side='left'), 0, 255).astype(np.uint8)
def up(a):
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
def maps(win, use2k):
    """Out-of-fold and test ensemble maps of one first-stage selection, exactly as in v15 / v15b."""
    oof = {k: load_oof(DIRS[k], 'oof2k_f' if (use2k and k == 'C') else 'oof_f') for k in win}
    if use2k:
        po = {s: ens([oof[k][s] if k == 'C' else up(oof[k][s]) for k in win]) for s in stems}
    else:
        po = {s: ens([oof[k][s] for k in win]) for s in stems}
    del oof
    h = np.zeros(256)
    for s in stems:
        h += np.bincount(po[s].ravel(), minlength=256)
    if use2k:
        tl = {k: np.load(f'{DIRS[k]}/test2k_mean.npz' if k == 'C' else f'{DIRS[k]}/test_mean.npz')['a'] for k in win}
        tm = np.stack([ens([tl[k][i] if k == 'C' else up(tl[k][i]) for k in win]) for i in range(len(tests))]); del tl
    else:
        tm = ens([np.load(f'{DIRS[k]}/test_mean.npz')['a'] for k in win])
    return po, qmatch(h, tm)[tm]
if DSP:
    SPO = load_oof(DSP, 'oofsp_f')
    hs = np.zeros(256)
    for s in stems:
        hs += np.bincount(SPO[s].ravel(), minlength=256)
    tsp = np.load(f'{DSP}/testsp_mean.npz')['a']
    SPT = qmatch(hs, tsp)[tsp]; del tsp
else:
    SPO, SPT = None, None
YY, XX = np.indices((H, W), dtype=np.float32)

def flatten(im):
    """Image divided by its radial median profile (limb-darkening removal), mapped [0.4, 1.4] -> [0, 255]."""
    a = im.astype(np.float32)
    small = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
    m = small > 20
    if m.sum() < 1000:
        return np.zeros(im.shape, np.uint8)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() * 4 + 1.5, xs.mean() * 4 + 1.5
    R = np.sqrt(m.sum() / np.pi) * 4
    yy, xx = np.indices(small.shape)
    rs = np.hypot(yy * 4 + 1.5 - cy, xx * 4 + 1.5 - cx) / R
    nb = 120
    b = np.clip((rs * nb).astype(int), 0, nb + 9)
    prof = np.full(nb + 10, np.nan, np.float32)
    for k in range(nb + 10):
        v = small[(b == k) & m]
        if v.size > 20:
            prof[k] = np.median(v)
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(nb + 10), np.where(ok)[0], prof[ok])
    r = np.hypot(YY - cy, XX - cx) / R
    pr = np.interp(r * nb, np.arange(nb + 10), prof)
    f = np.where((r < 1.0) & (a > 20), a / np.maximum(pr, 1.), 0.)
    return (np.clip((f - 0.4), 0, 1.) * 255).astype(np.uint8)

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

def build(job):
    """Crops of the selected instances of one image: list of (x [6, CH, CW] uint8, target or None, info)."""
    split, s = job
    cv2.setNumThreads(1)
    sel = CUR['sel'][split].get(s, [])
    if not sel:
        return split, s, []
    im = cv2.imread(TRP[s] if split == 'train' else TEP[s], cv2.IMREAD_GRAYSCALE)
    d = im[im > 20]; med = float(np.median(d)) if d.size > 1000 else 128.
    raw = np.clip(im.astype(np.float32) / med * 128, 0, 255).astype(np.uint8)
    fl = flatten(im)
    if split == 'train':
        p1, s1 = PO[s], SPO[s] if SPO is not None else None
    else:
        i = tests.index(s); p1, s1 = PT[i], SPT[i] if SPT is not None else None
    seeds = [mu.decode(dict(size=[H, W], counts=c.encode())) for c, _ in sel]
    allm = np.zeros((H, W), np.uint8)
    for k, m in enumerate(seeds):
        allm[m > 0] = k + 1
    out = []
    for k, (seed, (c, u)) in enumerate(zip(seeds, sel)):
        M = frame(seed)
        M1 = M.copy(); M1[:, :2] *= 2.         # maps at 1024 px: image x = 2 x_1024
        x = np.stack([warp(raw, M), warp(fl, M), warp(p1, M if p1.shape[0] == H else M1),
                      warp(s1, M1) if s1 is not None else np.zeros((CH, CW), np.uint8),
                      warp(seed * 255, M), warp(((allm > 0) & (allm != k + 1)).astype(np.uint8) * 255, M)])
        out.append((x, None, dict(stem=s, k=k, u=u, M=M)))
    return split, s, out

t0 = time.time()
PO, PT = maps(SD['winner'] if SD['winner'] in ('BCD', 'AB', 'ABCD') else 'BCD', bool(SD.get('use2k', True)))
INFO = {}
for v in VARS:
    CUR = dict(tag=v, sel=SELS[v])
    crops = {'train': [], 'test': []}
    with Pool(4) as pool:
        for n, (split, s, out) in enumerate(pool.imap_unordered(build, [('train', s) for s in stems] + [('test', s) for s in tests], chunksize=2)):
            crops[split] += out
            if n % 200 == 0:
                print(v, n, f'{time.time() - t0:.0f}s', flush=True)
    for split in crops:
        crops[split].sort(key=lambda r: (r[2]['stem'], r[2]['k']))
    np.save(f'/tmp/XTR_{v}.npy', np.stack([r[0] for r in crops['train']])); np.save(f'/tmp/XTE_{v}.npy', np.stack([r[0] for r in crops['test']]))
    INFO[v] = {sp: [r[2] for r in crops[sp]] for sp in crops}
    pickle.dump(dict(fold=[FOLD[i['stem']] for i in INFO[v]['train']]), open(f'/tmp/meta_{v}.pkl', 'wb'))
    pickle.dump(INFO[v], open(f'/kaggle/working/info_{v}.pkl', 'wb'))
    print(v, 'crops', len(INFO[v]['train']), len(INFO[v]['test']), f'{time.time() - t0:.0f}s', flush=True)
    del crops
del PO, PT

# %%
%%writefile /tmp/prev29.py
# Apply the saved fold models of a refiner to the crops of one selection: out-of-fold crops of fold f with the
# fold-f model, test crops with all five (mean).
import os, sys, pickle
import numpy as np, torch
import segmentation_models_pytorch as smp
var, tag, arch, enc, d = sys.argv[1:6]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FO = np.array(pickle.load(open(f'/tmp/meta_{var}.pkl', 'rb'))['fold'])
X = np.load(f'/tmp/XTR_{var}.npy', mmap_mode='r'); T = np.load(f'/tmp/XTE_{var}.npy', mmap_mode='r')

@torch.no_grad()
def predict(model, arr, idx):
    model.eval()
    res = np.zeros((len(idx), arr.shape[2], arr.shape[3]), np.uint8)
    for b in range(0, len(idx), 64):
        x = torch.from_numpy(np.array(arr[idx[b:b + 64]]).astype(np.float32) / 255.).to(DEV)
        acc = 0
        for fh in (False, True):
            for fv in (False, True):
                xx = x.flip(3) if fh else x
                xx = xx.flip(2) if fv else xx
                with torch.autocast(DEV, enabled=DEV == 'cuda'):
                    p = torch.sigmoid(model(xx)).float()
                p = p.flip(2) if fv else p
                acc = acc + (p.flip(3) if fh else p)
        res[b:b + 64] = (acc[:, 0] / 4 * 255).byte().cpu().numpy()
    return res

ro = np.zeros((len(X), X.shape[2], X.shape[3]), np.uint8); rt = np.zeros((len(T), T.shape[2], T.shape[3]), np.uint16)
for f in range(5):
    model = getattr(smp, arch)(enc, encoder_weights=None, in_channels=6, classes=1)
    model.load_state_dict(torch.load(f'{d}/refiner_f{f}.pt', map_location='cpu'))
    model.to(DEV)
    va = np.where(FO == f)[0]
    if len(va):
        ro[va] = predict(model, X, va)
    rt += predict(model, T, np.arange(len(T)))
    print(var, tag, 'fold', f, 'done', len(va), flush=True)
np.savez_compressed(f'/kaggle/working/ref_{var}_{tag}.npz', ro=ro, rt=((rt + 2) // 5).astype(np.uint8))

# %%
REF = {}
for tag, arch, enc in (('v17', 'UnetPlusPlus', 'resnet34'), ('v18', 'Unet', 'resnext50_32x4d'), ('v19', 'Unet', 'efficientnet-b4')):
    c = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/refiner_f0.pt', recursive=True) if tag in p and '/prev_' not in p]
    if c:
        REF[tag] = (arch, enc, sorted(c, key=len)[0])
print('refiners', REF, flush=True)
jobs = [(v, t) + REF[t] for v in VARS for t in REF]
chains = [' && '.join(f'{sys.executable} /tmp/prev29.py {v} {t} {a} {e} {d}' for v, t, a, e, d in jobs[g::2]) for g in (0, 1)]
ps = [subprocess.Popen(['bash', '-c', c or 'true'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)), stdout=open(f'/kaggle/working/log{g}.txt', 'w'),
                       stderr=subprocess.STDOUT) for g, c in enumerate(chains)]
t1 = time.time()
while any(p.poll() is None for p in ps):
    time.sleep(60)
    for g in (0, 1):
        ls = open(f'/kaggle/working/log{g}.txt').read().strip().split('\n')
        print(f'[{time.time() - t1:.0f}s gpu{g}]', ls[-1][:200], flush=True)
print('rc', [p.returncode for p in ps])
for g in (0, 1):
    print(open(f'/kaggle/working/log{g}.txt').read()[-1500:])

# %%
def annot(sid, s):
    sid = str(sid)
    return sid[:-(len(s) + 1)] if sid.endswith(s) and len(sid) > len(s) + 1 else sid.split('-')[0]
ANN = {s: [annot(sid, s) for sid, _ in GT[s]] for s in stems}
KG = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) for r in (1, 2, 3, 4)}
def paste(pred, M, t):
    Mi = cv2.invertAffineTransform(M)
    cs = np.array([[0, 0, 1], [CW, 0, 1], [0, CH, 1], [CW, CH, 1]], np.float64) @ Mi.T
    x0, y0 = np.clip(np.floor(cs.min(0)).astype(int), 0, W); x1, y1 = np.clip(np.ceil(cs.max(0)).astype(int) + 1, 0, W)
    Mb = M.copy(); Mb[:, 2] += M[:, :2] @ np.array([x0, y0], np.float64)
    pb = cv2.warpAffine(pred, Mb, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return pb >= t * 255, pb, x0, y0

def refine_seq(sel, preds, Ms, t, grow=0, collapse=0.5, use=None):
    """Pixel-disjoint refined outlines of one image in decreasing predicted quality; grow = dilation (+) or
    erosion (-) radius in pixels applied to every outline. Returns per instance (rle or None, mean probability)."""
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
        if grow > 0:
            m = cv2.dilate(m.astype(np.uint8), KG[grow]) > 0
        elif grow < 0:
            m = cv2.erode(m.astype(np.uint8), KG[-grow]) > 0
        h, w = m.shape
        m &= ~taken[y0:y0 + h, x0:x0 + w]
        if m.sum() < 50:
            continue
        taken[y0:y0 + h, x0:x0 + w] |= m
        full[y0:y0 + h, x0:x0 + w] = m
        res[k] = (mu.encode(full), pm)
        full[y0:y0 + h, x0:x0 + w] = 0
    return res

RO, RT, OFF, OFT, MS = {}, {}, {}, {}, {}
for v in VARS:
    acc_o = acc_t = 0; n = 0
    for t in REF:
        z = np.load(f'/kaggle/working/ref_{v}_{t}.npz'); acc_o = acc_o + z['ro'].astype(np.uint16); acc_t = acc_t + z['rt'].astype(np.uint16); n += 1
    RO[v] = ((acc_o + n // 2) // n).astype(np.uint8); RT[v] = ((acc_t + n // 2) // n).astype(np.uint8)
    for sp, O in (('train', OFF), ('test', OFT)):
        O[v] = {}; MS[(v, sp)] = defaultdict(list)
        for j, i in enumerate(INFO[v][sp]):
            O[v].setdefault(i['stem'], j); MS[(v, sp)][i['stem']].append(i['M'])
    print(v, 'refined crops', RO[v].shape, RT[v].shape, 'refiners', n, flush=True)

def image_pass(job):
    v, t, s = job
    cv2.setNumThreads(1)
    sel = SELS[v]['train'].get(s, [])
    if not sel:
        return job, None
    assert len(MS[(v, 'train')][s]) == len(sel)
    res = refine_seq(sel, RO[v][OFF[v][s]:OFF[v][s] + len(sel)], MS[(v, 'train')][s], t)
    kept = np.array([r is not None for r, _ in res]); pm = np.array([p for _, p in res])
    miou = np.zeros((len(sel), KOF[s]))
    rles = [r for r, _ in res if r is not None]; ki = np.where(kept)[0]
    for a, (_, g) in enumerate(GT[s]):
        if g and rles:
            iou = np.array(mu.iou(rles, g, [0] * len(g)))
            miou[ki, a] = np.where(iou > .5, iou, 0).max(1)
    return job, dict(u=np.array([q for _, q in sel]), kept=kept, pm=pm, miou=miou)
PASSES = [(v, t) for v in VARS for t in (0.35, 0.45)]
cv2.setNumThreads(0)
PRES = defaultdict(dict)
with Pool(4) as pool:
    for (v, t, s), r in pool.imap_unordered(image_pass, [p + (s,) for p in PASSES for s in stems], chunksize=8):
        if r is not None:
            PRES[(v, t)][s] = r
print('passes done', f'{time.time() - t0:.0f}s', flush=True)
P_SKIP = 4.66 / 6.75
RECS = [(s, i) for s in stems for i in range(KOF[s])]
def score(PR, lam, pmin=0.5, recs=None):
    num = 0.; tp = npk = ngt = 0
    for s, i in (RECS if recs is None else recs):
        g = len(GT[s][i][1]); ngt += g
        r = PR.get(s)
        if r is None:
            continue
        keep = r['kept'] & (r['u'] > lam) & (r['pm'] >= pmin)
        npk += int(keep.sum()); num += r['miou'][keep, i].sum(); tp += int((r['miou'][keep, i] > 0).sum())
    den = tp + .5 * (npk - tp) + .5 * (ngt - tp)
    return dict(pq=num / max(den, 1e-9), pq_us=P_SKIP * num / max(.5 * P_SKIP * ngt + .5 * npk, 1e-9), tp=tp, fp=npk - tp,
                fn=ngt - tp, sq=num / max(tp, 1))
GROUPS = {'all': RECS, 'K1': [r for r in RECS if KOF[r[0]] == 1], 'K2+': [r for r in RECS if KOF[r[0]] >= 2],
          'K3': [r for r in RECS if KOF[r[0]] == 3]}
LAMS = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42]
R = {}
for p in PASSES:
    for pmin in (0., 0.5):
        for lam in LAMS:
            R[p + (lam, pmin)] = {g: score(PRES[p], lam, pmin, recs) for g, recs in GROUPS.items()}
        for g in GROUPS:
            print(p, 'pmin', pmin, f'{g:4s}', ' '.join(f"{lam}:{R[p + (lam, pmin)][g]['pq']:.4f}" for lam in LAMS), flush=True)
BEST = {}
for v in VARS:
    for g in ('K2+', 'all'):
        k = max((k for k in R if k[0] == v), key=lambda k: R[k][g]['pq'])
        BEST[(v, g)] = k
        print('best', v, 'on', g, k, {a: {x: round(float(y), 4) for x, y in b.items()} for a, b in R[k].items()}, flush=True)
json.dump({'_'.join(map(str, k)): v for k, v in R.items()}, open('/kaggle/working/summary_v45.json', 'w'), indent=1, default=float)

# %%
def sub_rows(v, t, lam, pmin):
    rows = []
    for s in tests:
        sel = SELS[v]['test'].get(s, [])
        if not sel:
            continue
        preds = RT[v][OFT[v][s]:OFT[v][s] + len(sel)]; Ms = MS[(v, 'test')][s]
        use = np.array([q > lam for _, q in sel])
        if pmin > 0:
            pm = np.array([p for _, p in refine_seq(sel, preds, Ms, t, 0, use=use)])
            use &= pm >= pmin
        res = refine_seq(sel, preds, Ms, t, 0, use=use)
        rows += [(f'{s}_{k + 1}', r['counts'].decode()) for k, r in enumerate([r for r, _ in res if r is not None])]
    return pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
done = {}
for (v, g), (_, t, lam, pmin) in BEST.items():
    name = f'sub_{v}_best{g.replace("+", "p")}.csv'
    if (v, t, lam, pmin) not in done:
        done[(v, t, lam, pmin)] = sub_rows(v, t, lam, pmin)
    sub = done[(v, t, lam, pmin)]
    sub.to_csv(f'/kaggle/working/{name}', index=False)
    for s in tests[:40]:
        rs = [dict(size=[H, W], counts=c.encode()) for c in sub[sub.filament_id.str.startswith(s + '_')].segmentation_rle]
        if len(rs) > 1:
            ov = np.array(mu.iou(rs, rs, [0] * len(rs))); np.fill_diagonal(ov, 0)
            assert ov.max() == 0
    print(name, (v, t, lam, pmin), sub.shape, f'{len(sub) / len(tests):.2f} per image', flush=True)
bv = max(VARS, key=lambda v: R[BEST[(v, 'all')]]['all']['pq'])
print('winner', bv, BEST[(bv, 'all')], round(R[BEST[(bv, 'all')]]['all']['pq'], 4), flush=True)
shutil_src = f'/kaggle/working/sub_{bv}_bestall.csv'
pd.read_csv(shutil_src).to_csv('/kaggle/working/submission.csv', index=False)
print('done', f'{time.time() - t0:.0f}s', flush=True)
