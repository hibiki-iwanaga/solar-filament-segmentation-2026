# Training stage 03: source of the Kaggle notebook `filament-v12-flat-resnet34` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v12: third ensemble member with limb-darkening-flattened input (2 GPUs)
# Own pipeline. Same soft-label U-Net recipe as v5A (ResNet34, 1024 px crops of the 2048 px images,
# images with several annotators drawn twice as often), with two changes motivated by the filament
# literature (Diercke et al. 2024 and EdgeAttNet flatten the centre-to-limb variation of GONG H-alpha
# images before segmentation):
# - a second input channel: the image divided by its radial median brightness profile (disk centre and
#   radius fitted per image), clipped to [0.4, 1.4], so faint filaments near the limb have the same
#   contrast as at disk centre;
# - maps are stored at full 2048 px resolution (8-fold dihedral TTA), plus 1024 px copies.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.3.4'], check=False)

# %%
import os, glob, json, time, pickle
from collections import defaultdict
import numpy as np, cv2
from pycocotools import mask as mu
CFG = dict(full=True, enc='resnet34', epochs=28, reps=2, bs=6, lr=4e-4, multi_rep=2, seed=1, nfold=5, in_ch=2)
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
H = W = 2048
def ann_rle(a):
    r = mu.frPyObjects(a['segmentation'], H, W)
    return mu.merge(r) if isinstance(r, list) else r
GT = {s: [(sid, [ann_rle(a) for a in sets[sid]]) for sid in stem_sets[s]] for s in stems}
perm = np.random.RandomState(42).permutation(len(stems))
FOLD = {stems[i]: k % 5 for k, i in enumerate(perm)}
tests = sorted(os.path.splitext(f)[0] for f in os.listdir(TED))
S_ = H if CFG['full'] else 1024
X = np.zeros((len(stems), S_, S_), np.uint8); Y = np.zeros((len(stems), S_, S_), np.uint8)
T = np.zeros((len(tests), S_, S_), np.uint8)
XF = np.zeros((len(stems), S_, S_), np.uint8); TF = np.zeros((len(tests), S_, S_), np.uint8)
CEN = []
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

for i, s in enumerate(stems):
    im = cv2.imread(glob.glob(f'{TRD}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    acc = np.zeros((H, W), np.float32); cs = []
    for sid, rles in GT[s]:
        if rles:
            acc += mu.decode(rles).max(axis=2)
            cs += [mu.toBbox(r)[:2] + mu.toBbox(r)[2:] / 2 for r in rles]
    acc /= len(GT[s])
    X[i] = im if CFG['full'] else cv2.resize(im, (S_, S_), interpolation=cv2.INTER_AREA)
    XF[i] = flatten(im)
    Y[i] = (acc * 255).astype(np.uint8) if CFG['full'] else (cv2.resize(acc, (S_, S_), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    CEN.append(np.array(cs, np.float32).reshape(-1, 2) * (S_ / H))
for i, s in enumerate(tests):
    im = cv2.imread(glob.glob(f'{TED}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    T[i] = im if CFG['full'] else cv2.resize(im, (S_, S_), interpolation=cv2.INTER_AREA)
    TF[i] = flatten(im)
np.save('/tmp/X.npy', X); np.save('/tmp/Y.npy', Y); np.save('/tmp/T.npy', T)
np.save('/tmp/XF.npy', XF); np.save('/tmp/TF.npy', TF)
cv2.imwrite('/kaggle/working/flat_example.jpg', np.concatenate([X[0], XF[0]], 1)[::4, ::4])
print('flattened channel: mean on disk', float(XF[0][XF[0] > 0].mean()))
del XF, TF
pickle.dump(dict(stems=stems, tests=tests, fold=[FOLD[s] for s in stems], K=[len(GT[s]) for s in stems], cen=CEN),
            open('/tmp/meta.pkl', 'wb'))
del X, Y, T
print('cached', len(stems), len(tests), flush=True)

# %%
%%writefile /tmp/train5.py
import os, sys, json, time, pickle
import numpy as np, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
C = json.load(open('/tmp/cfg.json'))
folds = [int(f) for f in sys.argv[1].split(',')]
DEV = 'cuda'; CROP = 1024
meta = pickle.load(open('/tmp/meta.pkl', 'rb'))
FO = np.array(meta['fold']); KK = np.array(meta['K']); CEN = meta['cen']
X = np.load('/tmp/X.npy', mmap_mode='r'); Y = np.load('/tmp/Y.npy', mmap_mode='r'); T = np.load('/tmp/T.npy', mmap_mode='r')
XF = np.load('/tmp/XF.npy', mmap_mode='r'); TF = np.load('/tmp/TF.npy', mmap_mode='r')
S = X.shape[1]
def med_of(a):
    a = np.array(a); d = a[a > 20]
    return float(np.median(d)) if d.size > 1000 else 128.
MED = np.array([med_of(X[i]) for i in range(len(X))], np.float32)
MEDT = np.array([med_of(T[i]) for i in range(len(T))], np.float32)
def norm(x, med):
    return np.clip(x.astype(np.float32) / max(med, 1.) * 0.5, 0, 1.5)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx):
        rep = np.where(KK[idx] >= 2, C['multi_rep'], 1) * C['reps']
        self.idx = np.repeat(idx, rep)
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        if C['full']:
            c = CEN[j]
            if len(c) and np.random.rand() < .7:
                cx, cy = c[np.random.randint(len(c))] + np.random.uniform(-300, 300, 2)
            else:
                cx, cy = np.random.uniform(CROP / 2, S - CROP / 2, 2)
            x0 = int(np.clip(cx - CROP / 2, 0, S - CROP)); y0 = int(np.clip(cy - CROP / 2, 0, S - CROP))
            x = np.array(X[j, y0:y0 + CROP, x0:x0 + CROP]); y = np.array(Y[j, y0:y0 + CROP, x0:x0 + CROP])
            xf = np.array(XF[j, y0:y0 + CROP, x0:x0 + CROP])
        else:
            x, y, xf = np.array(X[j]), np.array(Y[j]), np.array(XF[j])
        R = x.shape[0]
        k = np.random.randint(4)
        x, y, xf = np.rot90(x, k), np.rot90(y, k), np.rot90(xf, k)
        if np.random.rand() < .5:
            x, y, xf = x[:, ::-1], y[:, ::-1], xf[:, ::-1]
        x, y, xf = np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(xf)
        if np.random.rand() < .5:
            M = cv2.getRotationMatrix2D((R / 2, R / 2), np.random.uniform(-180, 180), np.random.uniform(.9, 1.1))
            x = cv2.warpAffine(x, M, (R, R), flags=cv2.INTER_LINEAR)
            y = cv2.warpAffine(y, M, (R, R), flags=cv2.INTER_LINEAR)
            xf = cv2.warpAffine(xf, M, (R, R), flags=cv2.INTER_LINEAR)
        x = norm(x, MED[j])
        x = np.clip(x, 0, None) ** np.random.uniform(.8, 1.25) * np.random.uniform(.85, 1.15)
        xf = xf.astype(np.float32) / 255. * np.random.uniform(.9, 1.1)
        if np.random.rand() < .3:
            sg = np.random.uniform(.5, 1.5)
            x = cv2.GaussianBlur(x, (0, 0), sg); xf = cv2.GaussianBlur(xf, (0, 0), sg)
        if np.random.rand() < .3:
            x = x + np.random.normal(0, .02, x.shape).astype(np.float32)
        x = np.stack([x, xf]).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)[None].float() / 255.

def loss_fn(lo, y):
    bce = F.binary_cross_entropy_with_logits(lo, y)
    p = torch.sigmoid(lo)
    inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
    return bce + 1 - ((2 * inter + 1) / (den + 1)).mean()

@torch.no_grad()
def predict(model, arr, farr, meds, idx):
    """8-fold dihedral TTA at full resolution; returns (2048 px maps, 1024 px area-averaged maps)."""
    model.eval()
    out2 = np.zeros((len(idx), arr.shape[1], arr.shape[2]), np.uint8); out1 = np.zeros((len(idx), 1024, 1024), np.uint8)
    for n, j in enumerate(idx):
        x = np.stack([norm(np.array(arr[j]), meds[j]), np.array(farr[j]).astype(np.float32) / 255.])
        x = torch.from_numpy(x)[None].to(DEV)
        acc = 0
        for k in range(4):
            for fl in (False, True):
                xx = torch.rot90(x, k, (2, 3))
                if fl:
                    xx = xx.flip(3)
                with torch.autocast('cuda'):
                    p = torch.sigmoid(model(xx)).float()
                if fl:
                    p = p.flip(3)
                acc = acc + torch.rot90(p, -k, (2, 3))
        acc = acc / 8
        out2[n] = (acc[0, 0] * 255).byte().cpu().numpy()
        out1[n] = (F.interpolate(acc, size=(1024, 1024), mode='area')[0, 0] * 255).byte().cpu().numpy()
    return out2, out1

for f in folds:
    torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
    tr = np.where(FO != f)[0]; va = np.where(FO == f)[0]
    dl = torch.utils.data.DataLoader(DS(tr), batch_size=C['bs'], shuffle=True, num_workers=2, drop_last=True, pin_memory=True, persistent_workers=True)
    model = smp.Unet(C['enc'], encoder_weights='imagenet', in_channels=C['in_ch'], classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C['lr'], total_steps=C['epochs'] * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(C['epochs']):
        model.train(); t0 = time.time(); tl = 0
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast('cuda'):
                lo = model(x)
            loss = loss_fn(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tl += loss.item()
        print(f'fold {f} ep {ep} loss {tl / len(dl):.4f} {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'/kaggle/working/unet_f{f}.pt')
    o2, o1 = predict(model, X, XF, MED, va)
    np.savez_compressed(f'/kaggle/working/oof2k_f{f}.npz', a=o2); np.savez_compressed(f'/kaggle/working/oof_f{f}.npz', a=o1)
    del o2, o1
    t2, t1 = predict(model, T, TF, MEDT, np.arange(len(T)))
    np.savez_compressed(f'/tmp/test2k_f{f}.npz', a=t2); np.savez_compressed(f'/kaggle/working/test_f{f}.npz', a=t1)
    del t2, t1
    print(f'fold {f} done', flush=True)

# %%
import subprocess
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES='0'); env1 = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
p0 = subprocess.Popen([sys.executable, '/tmp/train5.py', '0,2,4'], env=env0, stdout=open('/kaggle/working/log0.txt', 'w'), stderr=subprocess.STDOUT)
p1 = subprocess.Popen([sys.executable, '/tmp/train5.py', '1,3'], env=env1, stdout=open('/kaggle/working/log1.txt', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while p0.poll() is None or p1.poll() is None:
    time.sleep(300)
    for k in (0, 1):
        ls = open(f'/kaggle/working/log{k}.txt').read().strip().split('\n')
        print(f'[{time.time() - t0:.0f}s gpu{k}]', ls[-1][:200], flush=True)
print('rc', p0.returncode, p1.returncode)
for k in (0, 1):
    print(open(f'/kaggle/working/log{k}.txt').read()[-1500:])
TPM = np.mean([np.load(f'/kaggle/working/test_f{f}.npz')['a'].astype(np.float32) for f in range(5)], 0).astype(np.uint8)
np.savez_compressed('/kaggle/working/test_mean.npz', a=TPM)
acc = None
for f in range(5):
    a = np.load(f'/tmp/test2k_f{f}.npz')['a'].astype(np.uint16)
    acc = a if acc is None else acc + a
    del a
np.savez_compressed('/kaggle/working/test2k_mean.npz', a=((acc + 2) // 5).astype(np.uint8))
print('saved', TPM.shape, acc.shape, sorted(os.listdir('/kaggle/working')))
