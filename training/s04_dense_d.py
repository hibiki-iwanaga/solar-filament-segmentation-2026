# Training stage 04: source of the Kaggle notebook `filament-v14-unetpp-spine` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v14: UNet++ member with a spine head (2 GPUs)
# Own pipeline. A fourth, architecturally different member of the soft-label ensemble:
# - UNet++ (ResNet34 encoder): nested dense skip pathways fuse boundary evidence with context, which suits
#   3-12 px wide ribbons whose boundary decides an IoU > 0.5 match;
# - input: the image and its limb-darkening-flattened version (radial median profile, as in v12), at 1024 px;
# - two outputs: the soft annotator-mean filament mask and the soft annotator-mean *spine* (the MAGFiLO spine
#   polylines, rasterised). The spine is one unbroken line per filament as the annotator saw it, so it carries
#   the grouping of fragments into filaments; its map becomes a descriptor for the candidate scorer;
# - loss: BCE + 0.5 soft Dice on the mask (Dice counteracts the background-dominated BCE on thin ribbons),
#   + 0.3 BCE on the spine; EMA of the weights; 512 px crops (70 % around filaments), 5 000 steps per fold;
# - out-of-fold and test maps at 1024 px with 8-fold dihedral TTA, for the mask and the spine.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.3.4'], check=False)

# %%
import os, glob, json, time, pickle
from collections import defaultdict
import numpy as np, cv2
from pycocotools import mask as mu
CFG = dict(arch='UnetPlusPlus', enc='resnet34', steps=5000, bs=12, crop=512, lr=4e-4, multi_rep=2, seed=3, ema=0.999,
           dice_w=0.5, spine_w=0.3)
json.dump(CFG, open('/tmp/cfg.json', 'w'))
ROOT = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026', recursive=True)[0]
TRJ = glob.glob(ROOT + '/train/*.json')[0]
TRD = ROOT + '/train/train_images'; TED = ROOT + '/test/test_images'
J = json.load(open(TRJ))
imgs = {im['id']: im for im in J['images']}
sets = defaultdict(list)
for a in J['annotations']:
    sets[a['image_id']].append(a)
stem_sets = defaultdict(list)
for sid, im in imgs.items():
    stem_sets[os.path.splitext(im['file_name'])[0]].append(sid)
stems = sorted(stem_sets)
H = W = 2048; S_ = 1024
def ann_rle(a):
    r = mu.frPyObjects(a['segmentation'], H, W)
    return mu.merge(r) if isinstance(r, list) else r
perm = np.random.RandomState(42).permutation(len(stems))
FOLD = {stems[i]: k % 5 for k, i in enumerate(perm)}
tests = sorted(os.path.splitext(f)[0] for f in os.listdir(TED))
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

def small(a, interp=cv2.INTER_AREA):
    return cv2.resize(a, (S_, S_), interpolation=interp)

X = np.zeros((len(stems), 2, S_, S_), np.uint8); Y = np.zeros((len(stems), 2, S_, S_), np.uint8)
T = np.zeros((len(tests), 2, S_, S_), np.uint8)
CEN = []
for i, s in enumerate(stems):
    im = cv2.imread(glob.glob(f'{TRD}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    X[i, 0] = small(im); X[i, 1] = small(flatten(im))
    acc = np.zeros((S_, S_), np.float32); sp = np.zeros((S_, S_), np.float32); cs = []
    for sid in stem_sets[s]:
        anns = sets[sid]
        if anns:
            acc += small(mu.decode([ann_rle(a) for a in anns]).max(axis=2).astype(np.float32))
            lines = np.zeros((S_, S_), np.uint8)
            for a in anns:
                pts = np.array(a['spine'], np.float32).reshape(-1, 2) * (S_ / H)
                if len(pts) >= 2:
                    cv2.polylines(lines, [np.round(pts * 4).astype(np.int32)], False, 1, thickness=3, lineType=cv2.LINE_8, shift=2)
                cs += [mu.toBbox(ann_rle(a))[:2] / 2 + mu.toBbox(ann_rle(a))[2:] / 4]
            sp += lines
    K = len(stem_sets[s])
    Y[i, 0] = np.clip(acc / K * 255, 0, 255).astype(np.uint8); Y[i, 1] = np.clip(sp / K * 255, 0, 255).astype(np.uint8)
    CEN.append(np.array(cs, np.float32).reshape(-1, 2))
for i, s in enumerate(tests):
    im = cv2.imread(glob.glob(f'{TED}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    T[i, 0] = small(im); T[i, 1] = small(flatten(im))
np.save('/tmp/X.npy', X); np.save('/tmp/Y.npy', Y); np.save('/tmp/T.npy', T)
pickle.dump(dict(stems=stems, tests=tests, fold=[FOLD[s] for s in stems], K=[len(stem_sets[s]) for s in stems], cen=CEN),
            open('/tmp/meta.pkl', 'wb'))
cv2.imwrite('/kaggle/working/example.jpg', np.concatenate([X[0, 0], X[0, 1], Y[0, 0], Y[0, 1]], 1)[::2, ::2])
print('cached', X.shape, T.shape, 'mean mask / spine target', float(Y[:, 0].mean()), float(Y[:, 1].mean()), flush=True)
del X, Y, T

# %%
%%writefile /tmp/train14.py
import os, sys, json, time, pickle, copy
import numpy as np, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
C = json.load(open('/tmp/cfg.json'))
folds = [int(f) for f in sys.argv[1].split(',')]
DEV = 'cuda'; CROP = C['crop']
meta = pickle.load(open('/tmp/meta.pkl', 'rb'))
FO = np.array(meta['fold']); KK = np.array(meta['K']); CEN = meta['cen']
X = np.load('/tmp/X.npy', mmap_mode='r'); Y = np.load('/tmp/Y.npy', mmap_mode='r'); T = np.load('/tmp/T.npy', mmap_mode='r')
S = X.shape[2]
def med_of(a):
    d = a[a > 20]
    return float(np.median(d)) if d.size > 1000 else 128.
MED = np.array([med_of(np.array(X[i, 0])) for i in range(len(X))], np.float32)
MEDT = np.array([med_of(np.array(T[i, 0])) for i in range(len(T))], np.float32)
def prep(x2, med):
    """uint8 [2, h, w] -> float [2, h, w]: image / median * 0.5 and flattened image in [0, 1]."""
    return np.stack([np.clip(x2[0].astype(np.float32) / max(med, 1.) * 0.5, 0, 1.5), x2[1].astype(np.float32) / 255.])

class DS(torch.utils.data.Dataset):
    def __init__(self, idx, n):
        w = np.where(KK[idx] >= 2, C['multi_rep'], 1).astype(float)
        self.idx = idx; self.p = w / w.sum(); self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        j = np.random.choice(self.idx, p=self.p)
        c = CEN[j]
        if len(c) and np.random.rand() < .7:
            cx, cy = c[np.random.randint(len(c))] + np.random.uniform(-150, 150, 2)
        else:
            cx, cy = np.random.uniform(CROP / 2, S - CROP / 2, 2)
        x0 = int(np.clip(cx - CROP / 2, 0, S - CROP)); y0 = int(np.clip(cy - CROP / 2, 0, S - CROP))
        x = np.array(X[j, :, y0:y0 + CROP, x0:x0 + CROP]); y = np.array(Y[j, :, y0:y0 + CROP, x0:x0 + CROP])
        k = np.random.randint(4)
        x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if np.random.rand() < .5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if np.random.rand() < .5:
            M = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), np.random.uniform(-180, 180), np.random.uniform(.9, 1.1))
            x = np.stack([cv2.warpAffine(c_, M, (CROP, CROP), flags=cv2.INTER_LINEAR) for c_ in x])
            y = np.stack([cv2.warpAffine(c_, M, (CROP, CROP), flags=cv2.INTER_LINEAR) for c_ in y])
        x = prep(x, MED[j])
        x[0] = np.clip(x[0], 0, None) ** np.random.uniform(.8, 1.25) * np.random.uniform(.85, 1.15)
        x[1] = x[1] * np.random.uniform(.9, 1.1)
        if np.random.rand() < .3:          # seeing
            sg = np.random.uniform(.5, 1.5)
            x = np.stack([cv2.GaussianBlur(c_, (0, 0), sg) for c_ in x])
        if np.random.rand() < .3:
            x[0] = x[0] + np.random.normal(0, .02, x[0].shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(y).float() / 255.

def loss_fn(lo, y):
    m_lo, s_lo = lo[:, :1].float(), lo[:, 1:2].float()
    ym, ys = y[:, :1], y[:, 1:2]
    bce = F.binary_cross_entropy_with_logits(m_lo, ym)
    p = torch.sigmoid(m_lo)
    inter = (p * ym).sum((2, 3)); den = (p + ym).sum((2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + C['dice_w'] * dice + C['spine_w'] * F.binary_cross_entropy_with_logits(s_lo, ys)

@torch.no_grad()
def predict(model, arr, meds, idx):
    model.eval()
    out = np.zeros((len(idx), 2, S, S), np.uint8)
    for n, j in enumerate(idx):
        x = torch.from_numpy(prep(np.array(arr[j]), meds[j]))[None].to(DEV)
        views = []
        for k in range(4):
            for fl in (False, True):
                xx = torch.rot90(x, k, (2, 3))
                views.append(xx.flip(3) if fl else xx)
        with torch.autocast('cuda'):
            p = torch.sigmoid(model(torch.cat(views))).float()
        acc = 0
        v = 0
        for k in range(4):
            for fl in (False, True):
                q = p[v:v + 1]
                if fl:
                    q = q.flip(3)
                acc = acc + torch.rot90(q, -k, (2, 3)); v += 1
        out[n] = (acc[0] / 8 * 255).byte().cpu().numpy()
    return out

for f in folds:
    torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
    tr = np.where(FO != f)[0]; va = np.where(FO == f)[0]
    dl = torch.utils.data.DataLoader(DS(tr, C['steps'] * C['bs']), batch_size=C['bs'], shuffle=False, num_workers=3,
                                     drop_last=True, pin_memory=True, persistent_workers=True,
                                     worker_init_fn=lambda w: np.random.seed(C['seed'] * 1000 + f * 100 + w + int(time.time()) % 1000))
    model = getattr(smp, C['arch'])(C['enc'], encoder_weights='imagenet', in_channels=2, classes=2).to(DEV)
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
    torch.save(ema.state_dict(), f'/kaggle/working/unetpp_f{f}.pt')
    o = predict(ema, X, MED, va)
    np.savez_compressed(f'/kaggle/working/oof_f{f}.npz', a=o[:, 0]); np.savez_compressed(f'/kaggle/working/oofsp_f{f}.npz', a=o[:, 1])
    t_ = predict(ema, T, MEDT, np.arange(len(T)))
    np.savez_compressed(f'/kaggle/working/test_f{f}.npz', a=t_[:, 0]); np.savez_compressed(f'/kaggle/working/testsp_f{f}.npz', a=t_[:, 1])
    print(f'fold {f} done {time.time() - t0:.0f}s', flush=True)

# %%
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES='0'); env1 = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
p0 = subprocess.Popen([sys.executable, '/tmp/train14.py', '0,2,4'], env=env0, stdout=open('/kaggle/working/log0.txt', 'w'), stderr=subprocess.STDOUT)
p1 = subprocess.Popen([sys.executable, '/tmp/train14.py', '1,3'], env=env1, stdout=open('/kaggle/working/log1.txt', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while p0.poll() is None or p1.poll() is None:
    time.sleep(120)
    for k in (0, 1):
        ls = open(f'/kaggle/working/log{k}.txt').read().strip().split('\n')
        print(f'[{time.time() - t0:.0f}s gpu{k}]', ls[-1][:200], flush=True)
print('rc', p0.returncode, p1.returncode)
for k in (0, 1):
    print(open(f'/kaggle/working/log{k}.txt').read()[-2500:])
for kind in ('test', 'testsp'):
    acc = np.mean([np.load(f'/kaggle/working/{kind}_f{f}.npz')['a'].astype(np.float32) for f in range(5)], 0)
    np.savez_compressed(f'/kaggle/working/{kind}_mean.npz' if kind == 'test' else '/kaggle/working/testsp_mean.npz',
                        a=acc.round().astype(np.uint8))
print('saved', sorted(os.listdir('/kaggle/working')))
