# Training stage 02: source of the Kaggle notebook `filament-v5b-effb4-5fold` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v5: 5-fold soft-label U-Net, final configuration (2 GPUs)
# Own pipeline. Produces out-of-fold and test probability maps for the candidate/selection stage (v4
# logic). Changes against v2: configuration chosen by the v3 comparison, and images labelled by
# several annotators are drawn twice as often (the test set is dominated by such images and their
# averaged label is the consensus that the metric rewards).

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.3.4'], check=False)

# %%
import os, glob, json, time, pickle
from collections import defaultdict
import numpy as np, cv2
from pycocotools import mask as mu
CFG = dict(full=False, enc='efficientnet-b4', epochs=30, reps=1, bs=4, lr=4e-4, multi_rep=2, seed=0, nfold=5)
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
CEN = []
for i, s in enumerate(stems):
    im = cv2.imread(glob.glob(f'{TRD}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    acc = np.zeros((H, W), np.float32); cs = []
    for sid, rles in GT[s]:
        if rles:
            acc += mu.decode(rles).max(axis=2)
            cs += [mu.toBbox(r)[:2] + mu.toBbox(r)[2:] / 2 for r in rles]
    acc /= len(GT[s])
    X[i] = im if CFG['full'] else cv2.resize(im, (S_, S_), interpolation=cv2.INTER_AREA)
    Y[i] = (acc * 255).astype(np.uint8) if CFG['full'] else (cv2.resize(acc, (S_, S_), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    CEN.append(np.array(cs, np.float32).reshape(-1, 2) * (S_ / H))
for i, s in enumerate(tests):
    im = cv2.imread(glob.glob(f'{TED}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    T[i] = im if CFG['full'] else cv2.resize(im, (S_, S_), interpolation=cv2.INTER_AREA)
np.save('/tmp/X.npy', X); np.save('/tmp/Y.npy', Y); np.save('/tmp/T.npy', T)
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
        else:
            x, y = np.array(X[j]), np.array(Y[j])
        R = x.shape[0]
        k = np.random.randint(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if np.random.rand() < .5:
            x, y = x[:, ::-1], y[:, ::-1]
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        if np.random.rand() < .5:
            M = cv2.getRotationMatrix2D((R / 2, R / 2), np.random.uniform(-180, 180), np.random.uniform(.9, 1.1))
            x = cv2.warpAffine(x, M, (R, R), flags=cv2.INTER_LINEAR)
            y = cv2.warpAffine(y, M, (R, R), flags=cv2.INTER_LINEAR)
        x = norm(x, MED[j])
        x = np.clip(x, 0, None) ** np.random.uniform(.8, 1.25) * np.random.uniform(.85, 1.15)
        if np.random.rand() < .3:
            x = cv2.GaussianBlur(x, (0, 0), np.random.uniform(.5, 1.5))
        if np.random.rand() < .3:
            x = x + np.random.normal(0, .02, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None], torch.from_numpy(y)[None].float() / 255.

def loss_fn(lo, y):
    bce = F.binary_cross_entropy_with_logits(lo, y)
    p = torch.sigmoid(lo)
    inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
    return bce + 1 - ((2 * inter + 1) / (den + 1)).mean()

@torch.no_grad()
def predict(model, arr, meds):
    model.eval(); out = np.zeros((len(arr), 1024, 1024), np.uint8)
    for n in range(len(arr)):
        x = torch.from_numpy(norm(np.array(arr[n]), meds[n]))[None, None].to(DEV)
        acc = 0
        for d in [None, (3,), (2,), (2, 3)]:
            with torch.autocast('cuda'):
                p = torch.sigmoid(model(x.flip(d) if d else x)).float()
            acc = acc + (p.flip(d) if d else p)
        acc = acc / 4
        if acc.shape[-1] != 1024:
            acc = F.interpolate(acc, size=(1024, 1024), mode='area')
        out[n] = (acc[0, 0] * 255).byte().cpu().numpy()
    return out

for f in folds:
    torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
    tr = np.where(FO != f)[0]; va = np.where(FO == f)[0]
    dl = torch.utils.data.DataLoader(DS(tr), batch_size=C['bs'], shuffle=True, num_workers=2, drop_last=True, pin_memory=True, persistent_workers=True)
    model = smp.Unet(C['enc'], encoder_weights='imagenet', in_channels=1, classes=1).to(DEV)
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
    np.savez_compressed(f'/kaggle/working/oof_f{f}.npz', a=predict(model, X[va], MED[va]))
    np.savez_compressed(f'/kaggle/working/test_f{f}.npz', a=predict(model, T, MEDT))
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
TPM = np.mean([np.load(f'/kaggle/working/test_f{f}.npz')['a'].astype(np.float32) for f in range(5)], 0).astype(np.uint8)
np.savez_compressed('/kaggle/working/test_mean.npz', a=TPM)
print('saved', TPM.shape)
