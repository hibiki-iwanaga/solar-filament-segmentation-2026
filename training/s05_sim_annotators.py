# Training stage 05: source of the Kaggle notebook `filament-v21-annotator-unet` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v21: annotator-conditioned U-Net (2 GPUs)
# Own pipeline, new first stage. The annotation of this task is a population of annotators with stable personal
# styles (measured in v20p on the images outlined by several annotators: number of filaments marked relative to
# the co-annotator sd 0.17 in log units, split-half correlation 0.73; outline area sd 0.12, correlation 0.86).
# 58 % of the training images were outlined by a single annotator, and those annotators are different people from
# the teams that outlined the multi-annotator images (the test set holds 2.44 annotation sets per image, i.e. it
# is team-annotated). A network trained on the pooled masks therefore absorbs personal style as label noise.
# This network is told who drew the mask:
# - U-Net (ResNet34), input = image + limb-darkening-flattened image at 1024 px (as v12 / v14);
# - an embedding of the annotator modulates every encoder stage and the decoder output (FiLM: scale and shift per
#   channel, initialised to the identity); 15 % of the samples use the "unknown annotator" embedding;
# - one training sample = one annotator record (its filament mask and its spines), so the target is never a
#   mixture of styles; loss BCE + 0.5 Dice (mask) + 0.3 BCE (spine), EMA, 512 px crops;
# - outputs for every out-of-fold and test image: the map of the unknown annotator, the mean map over the team
#   annotators and over all annotators, and one map per annotator (quantised at 0.3 / 0.5 / 0.7), from which the
#   decision layer can read how many simulated annotators would have drawn a candidate outline.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.3.4'], check=False)

# %%
import os, glob, json, time, pickle
from collections import defaultdict, Counter
import numpy as np, cv2
from pycocotools import mask as mu
CFG = dict(arch='Unet', enc='resnet34', steps=8000, bs=12, crop=512, lr=4e-4, seed=5, ema=0.999, dice_w=0.5, spine_w=0.3,
           emb=16, p_unknown=0.15, min_rec=8, min_team=15)
json.dump(CFG, open('/tmp/cfg21.json', 'w'))
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
def annot(sid, s):
    sid = str(sid)
    return sid[:-(len(s) + 1)] if sid.endswith(s) and len(sid) > len(s) + 1 else sid.split('-')[0]
RECS = [(i, sid, annot(sid, s)) for i, s in enumerate(stems) for sid in stem_sets[s]]     # in GT order of every image
n_rec = Counter(a for _, _, a in RECS)
n_team = Counter(a for i, _, a in RECS if len(stem_sets[stems[i]]) >= 2)
ANNS = sorted(a for a in n_rec if n_rec[a] >= CFG['min_rec'])           # index 1..A; 0 = unknown / rare annotator
AIDX = {a: k + 1 for k, a in enumerate(ANNS)}
TEAM = [AIDX[a] for a in ANNS if n_team[a] >= CFG['min_team']]
print('annotators', len(n_rec), 'kept', len(ANNS), 'team', [ANNS[k - 1] for k in TEAM], 'records', len(RECS),
      'records of rare annotators', sum(1 for _, _, a in RECS if a not in AIDX), flush=True)

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

t0 = time.time()
X = np.lib.format.open_memmap('/tmp/X.npy', 'w+', np.uint8, (len(stems), 2, S_, S_))
T = np.lib.format.open_memmap('/tmp/T.npy', 'w+', np.uint8, (len(tests), 2, S_, S_))
YR = np.lib.format.open_memmap('/tmp/YR.npy', 'w+', np.uint8, (len(RECS), 2, S_, S_))
CEN, NF = [], []
for i, s in enumerate(stems):
    im = cv2.imread(glob.glob(f'{TRD}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    X[i, 0] = small(im); X[i, 1] = small(flatten(im))
for r, (i, sid, a) in enumerate(RECS):
    anns = sets[sid]; cs = []
    if anns:
        YR[r, 0] = np.clip(small(mu.decode([ann_rle(x) for x in anns]).max(axis=2).astype(np.float32)) * 255, 0, 255).astype(np.uint8)
        lines = np.zeros((S_, S_), np.uint8)
        for x in anns:
            pts = np.array(x['spine'], np.float32).reshape(-1, 2) * (S_ / H)
            if len(pts) >= 2:
                cv2.polylines(lines, [np.round(pts * 4).astype(np.int32)], False, 255, thickness=3, lineType=cv2.LINE_8, shift=2)
            bb = mu.toBbox(ann_rle(x)); cs.append(bb[:2] / 2 + bb[2:] / 4)
        YR[r, 1] = lines
    CEN.append(np.array(cs, np.float32).reshape(-1, 2)); NF.append(len(anns))
for i, s in enumerate(tests):
    im = cv2.imread(glob.glob(f'{TED}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    T[i, 0] = small(im); T[i, 1] = small(flatten(im))
X.flush(); T.flush(); YR.flush()
pickle.dump(dict(stems=stems, tests=tests, fold=[FOLD[s] for s in stems], rec_img=[i for i, _, _ in RECS],
                 rec_ann=[AIDX.get(a, 0) for _, _, a in RECS], rec_name=[a for _, _, a in RECS], rec_n=NF, cen=CEN,
                 anns=ANNS, team=TEAM), open('/tmp/meta21.pkl', 'wb'))
pickle.dump(dict(anns=ANNS, team=TEAM, n_rec=dict(n_rec), n_team=dict(n_team)), open('/kaggle/working/annotators21.pkl', 'wb'))
cv2.imwrite('/kaggle/working/example.jpg', np.concatenate([X[0, 0], X[0, 1], YR[0, 0], YR[0, 1]], 1)[::2, ::2])
print('cached', X.shape, T.shape, YR.shape, 'mean mask / spine target', float(YR[:50, 0].mean()), float(YR[:50, 1].mean()),
      f'{time.time() - t0:.0f}s', flush=True)
del X, T, YR

# %%
%%writefile /tmp/train21.py
import os, sys, json, time, pickle, copy
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
C = json.load(open('/tmp/cfg21.json'))
OUT = os.environ.get('OUT21', '/kaggle/working')
folds = [int(f) for f in sys.argv[1].split(',')]
MODE = sys.argv[2] if len(sys.argv) > 2 else 'train'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; CROP = C['crop']
meta = pickle.load(open('/tmp/meta21.pkl', 'rb'))
FO = np.array(meta['fold']); RI = np.array(meta['rec_img']); RA = np.array(meta['rec_ann']); RN = np.array(meta['rec_n']); CEN = meta['cen']
A = len(meta['anns']); TEAM = np.array(meta['team'] if len(meta['team']) else list(range(1, len(meta['anns']) + 1)))
X = np.load('/tmp/X.npy', mmap_mode='r'); YR = np.load('/tmp/YR.npy', mmap_mode='r'); T = np.load('/tmp/T.npy', mmap_mode='r')
S = X.shape[2]
def med_of(a):
    d = a[a > 20]
    return float(np.median(d)) if d.size > 1000 else 128.
MED = np.array([med_of(np.array(X[i, 0])) for i in range(len(X))], np.float32)
MEDT = np.array([med_of(np.array(T[i, 0])) for i in range(len(T))], np.float32)
def prep(x2, med):
    """uint8 [2, h, w] -> float [2, h, w]: image / median * 0.5 and flattened image in [0, 1]."""
    return np.stack([np.clip(x2[0].astype(np.float32) / max(med, 1.) * 0.5, 0, 1.5), x2[1].astype(np.float32) / 255.])

class CondUnet(nn.Module):
    """U-Net whose encoder features and decoder output are modulated by an annotator embedding (index 0 = unknown)."""
    def __init__(self, n_ann, weights='imagenet'):
        super().__init__()
        self.net = getattr(smp, C['arch'])(C['enc'], encoder_weights=weights, in_channels=2, classes=2)
        chs = list(self.net.encoder.out_channels)[1:]
        dch = self.net.segmentation_head[0].in_channels
        self.emb = nn.Embedding(n_ann + 1, C['emb'])
        self.film = nn.ModuleList([nn.Linear(C['emb'], 2 * c) for c in chs] + [nn.Linear(C['emb'], 2 * dch)])
        for l in self.film:
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        nn.init.normal_(self.emb.weight, std=0.5)
        self.net.eval()
        try:                               # smp <= 0.3 takes the features unpacked, newer versions as one list
            with torch.no_grad():
                self.net.decoder(*[torch.zeros(1, c, 64 >> k, 64 >> k) for k, c in enumerate(self.net.encoder.out_channels)])
            self.star = True
        except TypeError:
            self.star = False
        self.net.train()
    @staticmethod
    def mod(f, gb):
        g, b = gb.chunk(2, 1)
        return f * (1 + g[:, :, None, None]).to(f.dtype) + b[:, :, None, None].to(f.dtype)
    def encode(self, x):
        return self.net.encoder(x)
    def decode(self, feats, a):
        e = self.emb(a)
        fs = [feats[0]] + [self.mod(f, l(e)) for f, l in zip(feats[1:], self.film[:-1])]
        d = self.net.decoder(*fs) if self.star else self.net.decoder(fs)
        return self.net.segmentation_head(self.mod(d, self.film[-1](e)))
    def forward(self, x, a):
        return self.decode(self.encode(x), a)

class DS(torch.utils.data.Dataset):
    def __init__(self, recs, n):
        self.recs = recs; self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        r = self.recs[np.random.randint(len(self.recs))]
        j = RI[r]; c = CEN[r]
        if len(c) and np.random.rand() < .7:
            cx, cy = c[np.random.randint(len(c))] + np.random.uniform(-150, 150, 2)
        else:
            cx, cy = np.random.uniform(CROP / 2, S - CROP / 2, 2)
        x0 = int(np.clip(cx - CROP / 2, 0, S - CROP)); y0 = int(np.clip(cy - CROP / 2, 0, S - CROP))
        x = np.array(X[j, :, y0:y0 + CROP, x0:x0 + CROP]); y = np.array(YR[r, :, y0:y0 + CROP, x0:x0 + CROP])
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
        a = 0 if np.random.rand() < C['p_unknown'] else int(RA[r])
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(y).float() / 255., a

def loss_fn(lo, y):
    m_lo, s_lo = lo[:, :1].float(), lo[:, 1:2].float()
    ym, ys = y[:, :1], y[:, 1:2]
    bce = F.binary_cross_entropy_with_logits(m_lo, ym)
    p = torch.sigmoid(m_lo)
    inter = (p * ym).sum((2, 3)); den = (p + ym).sum((2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + C['dice_w'] * dice + C['spine_w'] * F.binary_cross_entropy_with_logits(s_lo, ys)

AMP = DEV == 'cuda'; NW = int(os.environ.get('NW21', 3))
@torch.no_grad()
def predict(model, x2, med):
    """One image -> (unknown-annotator mask and spine [2, S, S] with 8 dihedral views, per-annotator mask [A, S, S]
    with 2 views), float32 on the device."""
    x = torch.from_numpy(prep(np.array(x2), med))[None].to(DEV)
    views = []
    for k in range(4):
        for fl in (False, True):
            xx = torch.rot90(x, k, (2, 3))
            views.append(xx.flip(3) if fl else xx)
    with torch.autocast(DEV, enabled=AMP):
        feats = model.encode(torch.cat(views))
        p = torch.sigmoid(model.decode(feats, torch.zeros(8, dtype=torch.long, device=DEV))).float()
    acc = 0; v = 0
    for k in range(4):
        for fl in (False, True):
            q = p[v:v + 1]
            if fl:
                q = q.flip(3)
            acc = acc + torch.rot90(q, -k, (2, 3)); v += 1
    p0 = acc[0] / 8
    pa = torch.zeros(A, S, S, device=DEV)
    ids = torch.arange(1, A + 1, device=DEV)
    for v in (0, 1):                                   # identity and horizontal flip
        fv = [f[v:v + 1] for f in feats]
        for ch in ids.split(8):
            with torch.autocast(DEV, enabled=AMP):
                q = torch.sigmoid(model.decode([f.expand(len(ch), -1, -1, -1) for f in fv], ch)[:, 0]).float()
            pa[ch - 1] += q.flip(2) if v == 1 else q
    return p0, pa / 2

def quant(pa):
    """[A, S, S] probabilities -> uint8 levels 0..3 (thresholds 0.3 / 0.5 / 0.7)."""
    return ((pa >= .3).to(torch.uint8) + (pa >= .5).to(torch.uint8) + (pa >= .7).to(torch.uint8)).cpu().numpy()

def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))

if MODE == 'train':
    for f in folds:
        torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
        tr = np.where(FO[RI] != f)[0]; va = np.where(FO == f)[0]
        dl = torch.utils.data.DataLoader(DS(tr, C['steps'] * C['bs']), batch_size=C['bs'], shuffle=False, num_workers=NW,
                                         drop_last=True, pin_memory=AMP, persistent_workers=NW > 0,
                                         worker_init_fn=lambda w: np.random.seed(C['seed'] * 1000 + f * 100 + w + int(time.time()) % 1000))
        model = CondUnet(A, os.environ.get('W21', 'imagenet') or None).to(DEV)
        ema = copy.deepcopy(model).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C['lr'], total_steps=C['steps'], pct_start=0.05)
        scaler = torch.amp.GradScaler(DEV, enabled=AMP)
        t0 = time.time(); tl = 0; step = 0
        for x, y, a in dl:
            model.train()
            x, y, a = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True), a.to(DEV)
            with torch.autocast(DEV, enabled=AMP):
                lo = model(x, a)
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
        del dl
        torch.save(ema.state_dict(), f'{OUT}/cond_f{f}.pt')
        # out-of-fold maps and a check that the embedding carries style: Dice of every held-out record against the
        # map of its own annotator, of the unknown annotator, of the team mean and of another (random) annotator
        ou = np.zeros((len(va), 2, S, S), np.uint8); om = np.zeros((len(va), S, S), np.uint8); oa = np.zeros((len(va), S, S), np.uint8)
        lev = np.lib.format.open_memmap(f'/tmp/lev_f{f}.npy', 'w+', np.uint8, (len(va), A, S, S))
        hist = np.zeros(256); st = []; rng = np.random.RandomState(f)
        for n, j in enumerate(va):
            p0, pa = predict(ema, X[j], MED[j])
            ou[n] = (p0 * 255).round().byte().cpu().numpy()
            om[n] = (pa[TEAM - 1].mean(0) * 255).round().byte().cpu().numpy(); oa[n] = (pa.mean(0) * 255).round().byte().cpu().numpy()
            lev[n] = quant(pa)
            hist += torch.bincount((pa * 255).round().long().flatten(), minlength=256).cpu().numpy()
            for r in np.where(RI == j)[0]:
                if RA[r] == 0 or RN[r] == 0:
                    continue
                g = torch.from_numpy(np.array(YR[r, 0]) >= 128).to(DEV)
                oth = int(rng.choice([k for k in range(1, A + 1) if k != RA[r]]))
                st.append(dict(a=int(RA[r]), own=dice((pa[RA[r] - 1] >= .5), g), unk=dice(p0[0] >= .5, g),
                               team=dice(pa[TEAM - 1].mean(0) >= .5, g), other=dice(pa[oth - 1] >= .5, g),
                               area_gt=int(g.sum()), area_own=int((pa[RA[r] - 1] >= .5).sum()), area_unk=int((p0[0] >= .5).sum())))
            if n % 40 == 0:
                print(f'fold {f} oof {n}/{len(va)} {time.time() - t0:.0f}s', flush=True)
        lev.flush()
        np.savez_compressed(f'{OUT}/oofu_f{f}.npz', a=ou[:, 0]); np.savez_compressed(f'{OUT}/oofsp_f{f}.npz', a=ou[:, 1])
        np.savez_compressed(f'{OUT}/oofm_f{f}.npz', a=om); np.savez_compressed(f'{OUT}/oofall_f{f}.npz', a=oa)
        np.savez_compressed(f'{OUT}/ooflev_f{f}.npz', a=np.load(f'/tmp/lev_f{f}.npy', mmap_mode='r'))
        os.remove(f'/tmp/lev_f{f}.npy')
        np.save(f'{OUT}/hist_f{f}.npy', hist); pickle.dump(st, open(f'{OUT}/style_f{f}.pkl', 'wb'))
        m = {k: float(np.mean([s[k] for s in st])) for k in ('own', 'unk', 'team', 'other')}
        ra = np.array([[np.log(max(s['area_own'], 1) / max(s['area_gt'], 1)), np.log(max(s['area_unk'], 1) / max(s['area_gt'], 1))] for s in st])
        print(f'fold {f} style check: mean Dice own {m["own"]:.4f} unknown {m["unk"]:.4f} team-mean {m["team"]:.4f} other {m["other"]:.4f}',
              f'| sd of log(area pred / area gt): own {ra[:, 0].std():.3f} unknown {ra[:, 1].std():.3f} | records {len(st)}',
              f'{time.time() - t0:.0f}s', flush=True)
        del ema, model
        if AMP:
            torch.cuda.empty_cache()
else:
    # test: the five fold models averaged per annotator, then quantised
    models = []
    for f in folds:
        m = CondUnet(A, None); m.load_state_dict(torch.load(f'{OUT}/cond_f{f}.pt', map_location='cpu')); models.append(m.to(DEV).eval())
    tu = np.zeros((len(T), 2, S, S), np.uint8)
    pa8 = np.lib.format.open_memmap('/tmp/test_pa.npy', 'w+', np.uint8, (len(T), A, S, S))
    t0 = time.time()
    for n in range(len(T)):
        p0 = 0; pa = 0
        for m in models:
            a, b = predict(m, T[n], MEDT[n]); p0 = p0 + a; pa = pa + b
        tu[n] = (p0 / len(models) * 255).round().byte().cpu().numpy(); pa8[n] = (pa / len(models) * 255).round().byte().cpu().numpy()
        if n % 30 == 0:
            print(f'test {n}/{len(T)} {time.time() - t0:.0f}s', flush=True)
    pa8.flush()
    np.savez_compressed(f'{OUT}/testu_mean.npz', a=tu[:, 0]); np.savez_compressed(f'{OUT}/testsp_mean.npz', a=tu[:, 1])
    print('test done', flush=True)

# %%
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES='0'); env1 = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
p0 = subprocess.Popen([sys.executable, '/tmp/train21.py', '0,2,4'], env=env0, stdout=open('/kaggle/working/log0.txt', 'w'), stderr=subprocess.STDOUT)
p1 = subprocess.Popen([sys.executable, '/tmp/train21.py', '1,3'], env=env1, stdout=open('/kaggle/working/log1.txt', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while p0.poll() is None or p1.poll() is None:
    time.sleep(120)
    for k in (0, 1):
        ls = open(f'/kaggle/working/log{k}.txt').read().strip().split('\n')
        print(f'[{time.time() - t0:.0f}s gpu{k}]', ls[-1][:200], flush=True)
print('rc', p0.returncode, p1.returncode)
for k in (0, 1):
    txt = open(f'/kaggle/working/log{k}.txt').read()
    print('\n'.join(l for l in txt.split('\n') if 'style check' in l or 'Error' in l or 'error' in l)); print(txt[-1500:])

# %%
# test maps: five fold models averaged per annotator; per-annotator maps quantile-matched to the out-of-fold value
# distribution (the average of five models is smoother than one model) before they are quantised
r = subprocess.run([sys.executable, '/tmp/train21.py', '0,1,2,3,4', 'test'], env=env0, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:])
A_ = len(ANNS)
pa8 = np.load('/tmp/test_pa.npy', mmap_mode='r')
h_oof = sum(np.load(f'/kaggle/working/hist_f{f}.npy') for f in range(5))
h_te = np.zeros(256)
for n in range(len(pa8)):
    h_te += np.bincount(pa8[n].ravel(), minlength=256)
c_oof = np.cumsum(h_oof) / h_oof.sum(); c_te = np.cumsum(h_te) / h_te.sum()
LUT = np.clip(np.searchsorted(c_oof, c_te, side='left'), 0, 255).astype(np.uint8)
print('quantile map of the test values at 64 / 128 / 192:', LUT[[64, 128, 192]].tolist(), flush=True)
lev = np.zeros(pa8.shape, np.uint8); tm = np.zeros((len(pa8), S_, S_), np.uint8); ta = np.zeros((len(pa8), S_, S_), np.uint8)
TI = np.array(TEAM) - 1
for n in range(len(pa8)):
    q = LUT[pa8[n]]
    lev[n] = (q >= 77).astype(np.uint8) + (q >= 128) + (q >= 179)
    tm[n] = q[TI].mean(0).round(); ta[n] = q.mean(0).round()
np.savez_compressed('/kaggle/working/testlev.npz', a=lev)
np.savez_compressed('/kaggle/working/testm_mean.npz', a=tm); np.savez_compressed('/kaggle/working/testall_mean.npz', a=ta)
np.save('/kaggle/working/lut_test.npy', LUT)
import pickle as _p
st = sum((_p.load(open(f'/kaggle/working/style_f{f}.pkl', 'rb')) for f in range(5)), [])
by = defaultdict(list)
for s in st:
    by[s['a']].append(s)
print('style check, all folds: Dice own / unknown / team-mean / other annotator',
      [round(float(np.mean([s[k] for s in st])), 4) for k in ('own', 'unk', 'team', 'other')], 'records', len(st), flush=True)
for a in sorted(by, key=lambda a: -len(by[a])):
    v = by[a]
    print(f'  {ANNS[a - 1]} n={len(v):3d} own {np.mean([s["own"] for s in v]):.3f} unknown {np.mean([s["unk"] for s in v]):.3f} '
          f'team {np.mean([s["team"] for s in v]):.3f} other {np.mean([s["other"] for s in v]):.3f} '
          f'area own/gt {np.exp(np.mean([np.log(max(s["area_own"], 1) / max(s["area_gt"], 1)) for s in v])):.2f} '
          f'unknown/gt {np.exp(np.mean([np.log(max(s["area_unk"], 1) / max(s["area_gt"], 1)) for s in v])):.2f}', flush=True)
print('saved', sorted(os.listdir('/kaggle/working')))
