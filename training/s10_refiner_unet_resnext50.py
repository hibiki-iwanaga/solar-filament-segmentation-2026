# Training stage 10: source of the Kaggle notebook `filament-v18-refiner-ens` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v18: second stage-2 refiner and refiner ensemble (2 GPUs)
# Own pipeline. v17 (UNet++ ResNet34 refiner trained on the seeds of two first-stage selections) raised the
# out-of-fold PQ of the v15b selection from 0.4413 to 0.4443. v18 trains a second refiner of a different design on
# exactly the same crops and averages it with v17:
# - U-Net with a ResNeXt50-32x4d encoder (v17: UNet++ with ResNet34), different random seed, 8 000 steps;
# - crops are rebuilt exactly as in v17 (same selections, same maps, same order); the alignment with the v17
#   out-of-fold and test predictions is checked before averaging;
# - the paste-back setting (threshold, collapse rule) and the choice between v17 alone, v18 alone and their mean
#   are made on the out-of-fold selection with the competition metric; the official self-evaluation overlap
#   counts (one-to-many / many-to-one) are reported for each.

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
SELPS = sorted(glob.glob('/kaggle/input/**/selected_v15.pkl', recursive=True))
SRC = {}
for pth in SELPS:
    d = pickle.load(open(pth, 'rb'))
    tag = 'v15b' if 'v15b' in pth else 'v15'
    SRC[tag] = dict(win=d['winner'], sel=d['sel'][d['winner']], use2k=bool(d.get('use2k', False)) and 'C' in d['winner'])
TARGET = 'v15b' if 'v15b' in SRC else sorted(SRC)[0]
D1K = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/oof_f0.npz', recursive=True))
def pick(key):
    c = [d for d in D1K if key in d]
    return c[0] if c else None
DIRS = {'A': pick('v5a'), 'B': pick('v5b'), 'C': pick('v12'), 'D': pick('v14')}
DIRS = {k: v for k, v in DIRS.items() if v}
DSP = DIRS.get('D')
print(ROOT, SELPS, {k: (v['win'], v['sel']['lam'], v['use2k']) for k, v in SRC.items()}, 'target', TARGET, DIRS, flush=True)
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
    if split == 'train':
        recs = [g for _, g in GT[s]]
        seed_rles = [dict(size=[H, W], counts=c.encode()) for c, _ in sel]
        ious = [np.array(mu.iou(seed_rles, g, [0] * len(g))) if g else np.zeros((len(sel), 0)) for g in recs]
    out = []
    for k, (seed, (c, u)) in enumerate(zip(seeds, sel)):
        M = frame(seed)
        M1 = M.copy(); M1[:, :2] *= 2.         # maps at 1024 px: image x = 2 x_1024
        x = np.stack([warp(raw, M), warp(fl, M), warp(p1, M if p1.shape[0] == H else M1),
                      warp(s1, M1) if s1 is not None else np.zeros((CH, CW), np.uint8),
                      warp(seed * 255, M), warp(((allm > 0) & (allm != k + 1)).astype(np.uint8) * 255, M)])
        tgt = None; nmatch = 0
        if split == 'train':
            acc = np.zeros((CH, CW), np.float32)
            for g, iou in zip(recs, ious):
                if iou.shape[1] == 0:
                    continue
                j = int(iou[k].argmax())
                if iou[k, j] > 0.2:
                    acc += warp(mu.decode(g[j]).astype(np.float32), M); nmatch += 1
            tgt = (np.clip(acc / max(nmatch, 1), 0, 1) * 255).astype(np.uint8)
        out.append((x, tgt, dict(stem=s, k=k, u=u, M=M, nmatch=nmatch, area=int(seed.sum()), src=CUR['tag'])))
    return split, s, out

t0 = time.time()
CROPS = {'train': [], 'test': []}
for tag in sorted(SRC, key=lambda k: k != TARGET):        # target selection first
    PO, PT = maps(SRC[tag]['win'], SRC[tag]['use2k'])
    CUR = dict(tag=tag, sel=SRC[tag]['sel'])
    jobs = [('train', s) for s in stems] + ([('test', s) for s in tests] if tag == TARGET else [])
    with Pool(4) as pool:
        for n, (split, s, out) in enumerate(pool.imap_unordered(build, jobs, chunksize=2)):
            CROPS[split] += out
            if n % 200 == 0:
                print(tag, n, f'{time.time() - t0:.0f}s', flush=True)
    del PO, PT
for split in CROPS:
    CROPS[split].sort(key=lambda r: (r[2]['src'] != TARGET, r[2]['stem'], r[2]['k']))
XTR = np.stack([r[0] for r in CROPS['train']]); YTR = np.stack([r[1] for r in CROPS['train']])
XTE = np.stack([r[0] for r in CROPS['test']])
INFO_TR = [r[2] for r in CROPS['train']]; INFO_TE = [r[2] for r in CROPS['test']]
del CROPS
np.save('/tmp/XTR.npy', XTR); np.save('/tmp/YTR.npy', YTR); np.save('/tmp/XTE.npy', XTE)
TGT_IDX = np.array([j for j, i in enumerate(INFO_TR) if i['src'] == TARGET])
pickle.dump(dict(fold=[FOLD[i['stem']] for i in INFO_TR], target=TGT_IDX), open('/tmp/meta16.pkl', 'wb'))
print('crops', XTR.shape, XTE.shape, {t: sum(i['src'] == t for i in INFO_TR) for t in SRC},
      'instances with a matching annotation:', float(np.mean([i['nmatch'] > 0 for i in INFO_TR])), f'{time.time() - t0:.0f}s', flush=True)
ex = np.concatenate([np.concatenate(list(XTR[j]) + [YTR[j]], 0) for j in range(0, min(len(XTR), 40), 8)], 1)
cv2.imwrite('/kaggle/working/example_crops.jpg', ex)
del XTR, YTR, XTE

# %%
CFG = dict(arch='Unet', enc='resnext50_32x4d', steps=8000, bs=32, lr=3e-4, ema=0.998, dice_w=0.5, seed=23)
json.dump(CFG, open('/tmp/cfg18.json', 'w'))

# %%
%%writefile /tmp/train18.py
import os, sys, json, time, pickle, copy
import numpy as np, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
C = json.load(open('/tmp/cfg18.json'))
folds = [int(f) for f in sys.argv[1].split(',')]
DEV = 'cuda'
meta = pickle.load(open('/tmp/meta16.pkl', 'rb'))
FO = np.array(meta['fold']); TG = np.zeros(len(FO), bool); TG[meta['target']] = True
X = np.load('/tmp/XTR.npy', mmap_mode='r'); Y = np.load('/tmp/YTR.npy', mmap_mode='r'); T = np.load('/tmp/XTE.npy', mmap_mode='r')
K3 = np.ones((3, 3), np.uint8)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx, n):
        self.idx = idx; self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        j = self.idx[np.random.randint(len(self.idx))]
        x = np.array(X[j]).astype(np.float32) / 255.; y = np.array(Y[j]).astype(np.float32) / 255.
        if np.random.rand() < .5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        if np.random.rand() < .5:
            x, y = x[:, ::-1], y[::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        r = np.random.rand()                       # the seed of a test instance is itself a noisy outline
        if r < .2:
            x[4] = cv2.dilate(x[4], K3, iterations=np.random.randint(1, 3))
        elif r < .4:
            x[4] = cv2.erode(x[4], K3, iterations=np.random.randint(1, 3))
        x[0] = x[0] * np.random.uniform(.85, 1.15); x[1] = x[1] * np.random.uniform(.9, 1.1)
        if np.random.rand() < .3:
            x[2] = np.clip(x[2] ** np.random.uniform(.8, 1.25), 0, 1)
        return torch.from_numpy(x), torch.from_numpy(y)[None]

def loss_fn(lo, y):
    lo = lo.float()
    bce = F.binary_cross_entropy_with_logits(lo, y)
    p = torch.sigmoid(lo)
    inter = (p * y).sum((1, 2, 3)); den = (p + y).sum((1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + C['dice_w'] * dice

@torch.no_grad()
def predict(model, arr, idx):
    model.eval()
    out = np.zeros((len(idx), arr.shape[2], arr.shape[3]), np.uint8)
    for b in range(0, len(idx), 64):
        x = torch.from_numpy(np.array(arr[idx[b:b + 64]]).astype(np.float32) / 255.).to(DEV)
        acc = 0
        for fh in (False, True):
            for fv in (False, True):
                xx = x.flip(3) if fh else x
                xx = xx.flip(2) if fv else xx
                with torch.autocast('cuda'):
                    p = torch.sigmoid(model(xx)).float()
                p = p.flip(2) if fv else p
                acc = acc + (p.flip(3) if fh else p)
        out[b:b + 64] = (acc[:, 0] / 4 * 255).byte().cpu().numpy()
    return out

for f in folds:
    torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
    tr = np.where(FO != f)[0]; va = np.where((FO == f) & TG)[0]
    dl = torch.utils.data.DataLoader(DS(tr, C['steps'] * C['bs']), batch_size=C['bs'], shuffle=False, num_workers=3,
                                     drop_last=True, pin_memory=True, persistent_workers=True,
                                     worker_init_fn=lambda w: np.random.seed(C['seed'] * 1000 + f * 100 + w + int(time.time()) % 1000))
    model = getattr(smp, C['arch'])(C['enc'], encoder_weights='imagenet', in_channels=6, classes=1).to(DEV)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C['lr'], total_steps=C['steps'], pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time(); tl = 0; step = 0
    for x, y in dl:
        model.train()
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        with torch.autocast('cuda'):
            lo = model(x)
        loss = loss_fn(lo, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(C['ema']).add_(pm.detach(), alpha=1 - C['ema'])
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        tl += loss.item(); step += 1
        if step % 500 == 0:
            print(f'fold {f} step {step} loss {tl / 500:.4f} {time.time() - t0:.0f}s', flush=True); tl = 0
    torch.save(ema.state_dict(), f'/kaggle/working/refiner_f{f}.pt')
    np.savez_compressed(f'/kaggle/working/roof_f{f}.npz', a=predict(ema, X, va), idx=va)
    np.savez_compressed(f'/kaggle/working/rtest_f{f}.npz', a=predict(ema, T, np.arange(len(T))))
    print(f'fold {f} done {time.time() - t0:.0f}s', flush=True)

# %%
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES='0'); env1 = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
p0 = subprocess.Popen([sys.executable, '/tmp/train18.py', '0,2,4'], env=env0, stdout=open('/kaggle/working/log0.txt', 'w'), stderr=subprocess.STDOUT)
p1 = subprocess.Popen([sys.executable, '/tmp/train18.py', '1,3'], env=env1, stdout=open('/kaggle/working/log1.txt', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while p0.poll() is None or p1.poll() is None:
    time.sleep(60)
    for k in (0, 1):
        ls = open(f'/kaggle/working/log{k}.txt').read().strip().split('\n')
        print(f'[{time.time() - t0:.0f}s gpu{k}]', ls[-1][:200], flush=True)
print('rc', p0.returncode, p1.returncode)
for k in (0, 1):
    print(open(f'/kaggle/working/log{k}.txt').read()[-2500:])

# %% [markdown]
# ## Paste back and measure on the out-of-fold selection

# %%
def load_refined(d):
    """Out-of-fold (target crops) and test refined probabilities saved by a refiner notebook in directory d."""
    ro = np.zeros((len(INFO_TR), CH, CW), np.uint8); idx_all = []
    for f in range(5):
        z = np.load(f'{d}/roof_f{f}.npz'); ro[z['idx']] = z['a']; idx_all.append(np.sort(z['idx']))
    rt = np.mean([np.load(f'{d}/rtest_f{f}.npz')['a'].astype(np.float32) for f in range(5)], 0).round().astype(np.uint8)
    return ro, rt, np.concatenate(idx_all)
RO18, RT18, IDX18 = load_refined('/kaggle/working')
D17 = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/roof_f0.npz', recursive=True) if 'v17' in p]
VARIANTS = {'v18': (RO18, RT18)}
if D17:
    RO17, RT17, IDX17 = load_refined(D17[0])
    aligned = (RT17.shape == RT18.shape) and np.array_equal(np.sort(IDX17), np.sort(IDX18))
    print('v17 outputs', D17[0], 'aligned with the rebuilt crops:', aligned, RT17.shape, RT18.shape, flush=True)
    if aligned:
        VARIANTS['v17'] = (RO17, RT17)
        VARIANTS['mean'] = (((RO17.astype(np.uint16) + RO18 + 1) // 2).astype(np.uint8),
                            ((RT17.astype(np.uint16) + RT18 + 1) // 2).astype(np.uint8))
SEL = SRC[TARGET]['sel']

def paste(pred, M, t):
    """Refined crop probability -> boolean mask at 2048 px (bounding box, offset)."""
    Mi = cv2.invertAffineTransform(M)
    cs = np.array([[0, 0, 1], [CW, 0, 1], [0, CH, 1], [CW, CH, 1]], np.float64) @ Mi.T
    x0, y0 = np.clip(np.floor(cs.min(0)).astype(int), 0, W); x1, y1 = np.clip(np.ceil(cs.max(0)).astype(int) + 1, 0, W)
    Mb = M.copy(); Mb[:, 2] += M[:, :2] @ np.array([x0, y0], np.float64)
    pb = cv2.warpAffine(pred, Mb, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return pb >= t * 255, x0, y0

def refine_image(sel, preds, infos, t, mode, collapse=0.3):
    """Refined, pixel-disjoint RLEs of one image. mode: 'seed' (no refinement), 'fallback', 'drop'.
    An instance whose refined outline keeps less than `collapse` of the seed area is dropped ('drop') or kept as the
    seed ('fallback')."""
    seeds = [mu.decode(dict(size=[H, W], counts=c.encode())).astype(bool) for c, _ in sel]
    order = np.argsort([-u for _, u in sel])
    taken = np.zeros((H, W), bool); res = [None] * len(sel)
    for k in order:
        m = np.zeros((H, W), bool, order='F')
        if mode == 'seed':
            m |= seeds[k]
        else:
            pb, x0, y0 = paste(preds[k], infos[k]['M'], t)
            sb = seeds[k][y0:y0 + pb.shape[0], x0:x0 + pb.shape[1]]
            n, lab = cv2.connectedComponents(pb.astype(np.uint8), connectivity=8)
            keep = np.unique(lab[pb & sb]); keep = keep[keep > 0]
            if len(keep):
                m[y0:y0 + pb.shape[0], x0:x0 + pb.shape[1]] = np.isin(lab, keep)
            if m.sum() < collapse * seeds[k].sum():
                if mode == 'drop':
                    continue
                m[:] = seeds[k]
        m &= ~taken
        if m.sum() < 50:
            continue
        taken |= m
        res[k] = mu.encode(m.astype(np.uint8, order='F'))
    return [r for r in res if r is not None]

BY_STEM = defaultdict(list)
for j, i in enumerate(INFO_TR):
    if i['src'] == TARGET:
        BY_STEM[i['stem']].append(j)
def evaluate(var, t, mode, collapse):
    """Pooled PQ on the out-of-fold target selection, plus the official self-evaluation overlap counts
    (IoU > 0): ground-truth filaments touched by several predictions (1:n, n>1) and predictions touching
    several ground-truth filaments (n:1, n>1)."""
    RO = VARIANTS[var][0]
    num = den = 0.; one_many = many_one = n_gt = n_pred = n_out = 0
    for s in stems:
        den += .5 * NGT[s]
        js = BY_STEM.get(s, [])
        if not js:
            continue
        rles = refine_image(SEL['train'][s], [RO[j] for j in js], [INFO_TR[j] for j in js], t, mode, collapse)
        den += .5 * KOF[s] * len(rles); n_out += len(rles)
        for _, g in GT[s]:
            if g and rles:
                iou = np.array(mu.iou(rles, g, [0] * len(g)))
                num += np.where(iou > .5, iou, 0).max(1).sum()
                touch = iou > 0
                one_many += int((touch.sum(0) > 1).sum()); many_one += int((touch.sum(1) > 1).sum())
                n_gt += len(g); n_pred += len(rles)
    return dict(pq=num / den, one_many=one_many / max(n_gt, 1), many_one=many_one / max(n_pred, 1),
                per_img=n_out / len(stems))

def ev_job(a):
    return a, evaluate(*a)
GRID = [('v18', 0.5, 'seed', 0.3)] + [(v, t, 'drop', 0.5) for v in VARIANTS for t in (0.35, 0.45)]
cv2.setNumThreads(0)
with Pool(4) as pool:
    R = dict(pool.map(ev_job, GRID))
for k, v in R.items():
    print('OOF', k, {a: round(b, 4) for a, b in v.items()}, flush=True)
best = max(R, key=lambda k: R[k]['pq'])
print('best', best, R[best], 'vs seeds', R[('v18', 0.5, 'seed', 0.3)], flush=True)

# %%
BY_T = defaultdict(list)
for j, i in enumerate(INFO_TE):
    BY_T[i['stem']].append(j)
def sub_rows(var, t, mode, collapse=0.3):
    RT = VARIANTS[var][1]
    rows = []
    for s in tests:
        js = BY_T.get(s, [])
        if not js:
            continue
        rles = refine_image(SEL['test'][s], [RT[j] for j in js], [INFO_TE[j] for j in js], t, mode, collapse)
        rows += [(f'{s}_{k + 1}', r['counts'].decode()) for k, r in enumerate(rles)]
    return pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
sub = sub_rows(*best)
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('submission', best, sub.shape, f'{len(sub) / len(tests):.2f} per image', flush=True)
for v in VARIANTS:
    kv = max((k for k in R if k[0] == v and k[2] != 'seed'), key=lambda k: R[k]['pq'])
    sub_rows(*kv).to_csv(f'/kaggle/working/submission_{v}.csv', index=False)
json.dump({'_'.join(map(str, k)): v for k, v in R.items()}, open('/kaggle/working/summary_v18.json', 'w'), indent=1)
