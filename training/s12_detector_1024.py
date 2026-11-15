# Training stage 12: source of the Kaggle notebook `filament-v31-maskrcnn2-views` (unchanged; run it on Kaggle or with tools/run_stage.py).

# %% [markdown]
# # Solar filaments v31: a second instance detector and test-time views for the detector evidence (2 GPUs)
# Own pipeline. v27 trained a Mask R-CNN (ResNet50-FPN) whose instances, turned into candidate descriptors, lifted the
# decision layer by +0.007 PQ out of fold. Two ways to make that evidence less noisy are prepared here:
# 1. a second detector of the same family but a different design (torchvision Mask R-CNN v2: ResNet50-FPN with the
#    improved heads, COCO weights, another seed), trained on the same five folds;
# 2. four views (identity, horizontal flip, vertical flip, 90 degree rotation) of every image for both detectors, the
#    detections mapped back to the image frame. The descriptors are averaged over views and models downstream.
# Inference uses the provided test images only; no file-name metadata is used.

# %%
import os, glob, json, time, pickle, subprocess, sys
from collections import defaultdict
import numpy as np, cv2
from pycocotools import mask as mu
CFG = dict(steps=6000, bs=2, lr=0.01, warm=500, seed=11, score_min=0.05, max_det=100)
P27 = glob.glob('/kaggle/input/**/det_f0.pt', recursive=True)
CFG['d27'] = os.path.dirname(P27[0]) if P27 else ''
print('v27 weights', CFG['d27'], flush=True)
json.dump(CFG, open('/tmp/cfg27.json', 'w'))
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

t0 = time.time()
X = np.lib.format.open_memmap('/tmp/X27.npy', 'w+', np.uint8, (len(stems), 2, S_, S_))
T = np.lib.format.open_memmap('/tmp/T27.npy', 'w+', np.uint8, (len(tests), 2, S_, S_))
RECS = []          # per image: list of records, each a list of instance RLEs at 1024 px (2 x 2 max pooling keeps thin ribbons)
for i, s in enumerate(stems):
    im = cv2.imread(glob.glob(f'{TRD}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    X[i, 0] = small(im); X[i, 1] = small(flatten(im))
    recs = []
    for sid in stem_sets[s]:
        inst = []
        for a in sets[sid]:
            m = mu.decode(ann_rle(a)).reshape(S_, 2, S_, 2).max((1, 3))
            if m.sum() >= 12:
                inst.append(mu.encode(np.asfortranarray(m)))
        recs.append(inst)
    RECS.append(recs)
for i, s in enumerate(tests):
    im = cv2.imread(glob.glob(f'{TED}/{s}.*')[0], cv2.IMREAD_GRAYSCALE)
    T[i, 0] = small(im); T[i, 1] = small(flatten(im))
X.flush(); T.flush()
pickle.dump(dict(stems=stems, tests=tests, fold=[FOLD[s] for s in stems], recs=RECS), open('/tmp/meta27.pkl', 'wb'))
print('cached', X.shape, T.shape, 'records', sum(len(r) for r in RECS), 'instances', sum(len(x) for r in RECS for x in r),
      'mean instances per record', float(np.mean([len(x) for r in RECS for x in r])), f'{time.time() - t0:.0f}s', flush=True)
del X, T

# %%
%%writefile /tmp/train27.py
import os, sys, json, time, pickle, math
import numpy as np, cv2, torch, torchvision
from pycocotools import mask as mu
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
C = json.load(open('/tmp/cfg27.json'))
OUT = os.environ.get('OUT27', '/kaggle/working')
NW = int(os.environ.get('NW27', '2'))
PRE = os.environ.get('W27', 'coco')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M = pickle.load(open('/tmp/meta27.pkl', 'rb'))
stems, tests, fold, RECS = M['stems'], M['tests'], np.array(M['fold']), M['recs']
X = np.load('/tmp/X27.npy', mmap_mode='r'); T = np.load('/tmp/T27.npy', mmap_mode='r')
S_ = X.shape[-1]

def to_input(x2):
    """[2, S, S] uint8 (raw, flat field) -> [3, S, S] float in [0, 1]."""
    a = torch.from_numpy(np.ascontiguousarray(x2)).float() / 255.
    return torch.stack([a[0], a[1], a[1]])

class DS(torch.utils.data.Dataset):
    def __init__(self, idx, n):
        self.idx, self.n = idx, n
    def __len__(self):
        return self.n
    def __getitem__(self, k):
        rng = np.random.default_rng((C['seed'], k, os.getpid(), int(time.time() * 1000) % 100000))
        i = int(self.idx[rng.integers(len(self.idx))])
        recs = RECS[i]
        inst = recs[int(rng.integers(len(recs)))] if recs else []
        x = np.array(X[i])
        ms = mu.decode(inst).transpose(2, 0, 1) if inst else np.zeros((0, S_, S_), np.uint8)
        kr = int(rng.integers(4))
        if kr:
            x = np.rot90(x, kr, (1, 2)); ms = np.rot90(ms, kr, (1, 2))
        if rng.random() < .5:
            x = x[:, :, ::-1]; ms = ms[:, :, ::-1]
        g = rng.uniform(0.85, 1.15); b = rng.uniform(-12, 12)
        x = np.clip(x.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
        boxes, keep = [], []
        for j, m in enumerate(ms):
            ys, xs = np.nonzero(m)
            if len(xs) < 12:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append([x0, y0, x1, y1]); keep.append(j)
        ms = np.ascontiguousarray(ms[keep]) if keep else np.zeros((0, S_, S_), np.uint8)
        tgt = dict(boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), labels=torch.ones(len(boxes), dtype=torch.int64),
                   masks=torch.from_numpy(ms))
        return to_input(x), tgt

def build(arch='v2', pre=None):
    kw = dict(min_size=S_, max_size=S_, box_detections_per_img=C['max_det'], box_score_thresh=C['score_min'],
              image_mean=[0.45, 0.45, 0.45], image_std=[0.22, 0.22, 0.22])
    fn = maskrcnn_resnet50_fpn_v2 if arch == 'v2' else maskrcnn_resnet50_fpn
    if (PRE if pre is None else pre) == 'coco':
        model = fn(weights='DEFAULT', **kw)
    else:
        model = fn(weights=None, weights_backbone=None, **kw)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return model

VIEWS = ((0, False), (0, True), (2, True), (1, False))      # (quarter turns, horizontal flip): identity, h-flip, v-flip, 90 degrees
def fwd(a, view):
    k, fl = view
    a = np.rot90(a, k, (-2, -1)) if k else a
    return a[..., ::-1] if fl else a
def inv(a, view):
    k, fl = view
    a = a[..., ::-1] if fl else a
    return np.rot90(a, -k, (-2, -1)) if k else a
@torch.no_grad()
def detect(model, x2, view=(0, False)):
    """Instances of one image seen in one view, mapped back to the image frame: [(RLE counts at 2048 px, score)]."""
    model.eval()
    with torch.autocast('cuda', enabled=DEV == 'cuda'):
        o = model([to_input(np.ascontiguousarray(fwd(x2, view))).to(DEV)])[0]
    out = []
    sc = o['scores'].float().cpu().numpy(); mk = o['masks'][:, 0].float().cpu().numpy()
    for s, m in zip(sc, mk):
        m = np.ascontiguousarray(inv(m, view))
        big = cv2.resize(m, (2 * S_, 2 * S_), interpolation=cv2.INTER_LINEAR) >= 0.5
        if big.sum() < 40:
            continue
        out.append((mu.encode(np.asfortranarray(big.astype(np.uint8)))['counts'].decode(), float(s)))
    return out

def run(f):
    torch.manual_seed(C['seed'] + f); np.random.seed(C['seed'] + f)
    tr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    model = build().to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=C['lr'], momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / C['warm']) * 0.5 * (1 + math.cos(math.pi * min(s, C['steps']) / C['steps'])))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == 'cuda')
    dl = torch.utils.data.DataLoader(DS(tr, C['steps'] * C['bs']), batch_size=C['bs'], shuffle=False, num_workers=NW,
                                     collate_fn=lambda b: tuple(zip(*b)), drop_last=True, persistent_workers=NW > 0)
    t0 = time.time(); run_l = {}
    model.train()
    for step, (xs, ts) in enumerate(dl, 1):
        xs = [x.to(DEV) for x in xs]; ts = [{k: v.to(DEV) for k, v in t.items()} for t in ts]
        with torch.autocast('cuda', enabled=DEV == 'cuda'):
            ld = model(xs, ts)
        loss = sum(ld.values())
        if not torch.isfinite(loss):
            print('non-finite loss, step skipped', step, flush=True); opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.)
        scaler.step(opt); scaler.update(); sched.step()
        for k, v in ld.items():
            run_l[k] = 0.98 * run_l.get(k, float(v.detach())) + 0.02 * float(v.detach())
        if step % 500 == 0 or step == C['steps']:
            print(f'fold {f} step {step}', {k.replace('loss_', ''): round(v, 3) for k, v in run_l.items()}, f'{time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{OUT}/det2_f{f}.pt')
    infer(model, f, 'm31', t0)
    del model; torch.cuda.empty_cache()

def infer(model, f, tag, t0):
    va = np.where(fold == f)[0]
    oofv = {stems[i]: [detect(model, np.array(X[i]), v) for v in VIEWS] for i in va}
    pickle.dump(oofv, open(f'{OUT}/{tag}_oof_f{f}.pkl', 'wb'))
    oof = {s: v[0] for s, v in oofv.items()}
    # sanity: how many instances of the first record are found (IoU > 0.5) by detections with score >= 0.5
    hit = tot = npred = 0
    for i in va:
        dets = [dict(size=[2 * S_, 2 * S_], counts=c.encode()) for c, s in oof[stems[i]] if s >= 0.5]
        npred += len(dets)
        if not RECS[i] or not RECS[i][0]:
            continue
        gts = [mu.encode(np.asfortranarray(np.repeat(np.repeat(mu.decode(g), 2, 0), 2, 1))) for g in RECS[i][0]]
        tot += len(gts)
        if dets:
            hit += int((np.array(mu.iou(dets, gts, [0] * len(gts))) > .5).any(0).sum())
    print(f'{tag} fold {f} detector check: instances of the first record found {hit}/{tot} = {hit / max(tot, 1):.3f} | detections with score >= 0.5 per image {npred / len(va):.2f}',
          f'| all detections per image {np.mean([len(v) for v in oof.values()]):.1f} {time.time() - t0:.0f}s', flush=True)
    te = {tests[i]: [detect(model, np.array(T[i]), v) for v in VIEWS] for i in range(len(tests))}
    pickle.dump(te, open(f'{OUT}/{tag}_test_f{f}.pkl', 'wb'))
    print(f'{tag} fold {f} test done, detections per image and view {np.mean([len(x) for v in te.values() for x in v]):.1f} | agreement of views: share of',
          f'identity detections (score >= 0.5) matched by the h-flip view at IoU > 0.5: {view_agree(te):.3f} {time.time() - t0:.0f}s', flush=True)

def view_agree(te):
    hit = tot = 0
    for v in te.values():
        a = [dict(size=[2 * S_, 2 * S_], counts=c.encode()) for c, s in v[0] if s >= 0.5]
        b = [dict(size=[2 * S_, 2 * S_], counts=c.encode()) for c, s in v[1] if s >= 0.3]
        tot += len(a)
        if a and b:
            hit += int((np.array(mu.iou(a, b, [0] * len(b))) > .5).any(1).sum())
    return hit / max(tot, 1)

for f in [int(v) for v in sys.argv[1].split(',')]:
    run(f)
if len(sys.argv) > 2 and C['d27']:           # views for the saved v27 detectors (first design)
    for f in [int(v) for v in sys.argv[2].split(',')]:
        t0 = time.time()
        model = build('v1', 'none')
        model.load_state_dict(torch.load(f"{C['d27']}/det_f{f}.pt", map_location='cpu')); model.to(DEV)
        infer(model, f, 'm27', t0)
        del model; torch.cuda.empty_cache()

# %%
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES='0'); env1 = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
p0 = subprocess.Popen([sys.executable, '/tmp/train27.py', '0,2,4'], env=env0, stdout=open('/kaggle/working/log0.txt', 'w'), stderr=subprocess.STDOUT)
p1 = subprocess.Popen([sys.executable, '/tmp/train27.py', '1,3', '0,1,2,3,4'], env=env1, stdout=open('/kaggle/working/log1.txt', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while p0.poll() is None or p1.poll() is None:
    time.sleep(120)
    for k in (0, 1):
        ls = open(f'/kaggle/working/log{k}.txt').read().strip().split('\n')
        print(f'[{time.time() - t0:.0f}s gpu{k}]', ls[-1][:200], flush=True)
print('rc', p0.returncode, p1.returncode)
for k in (0, 1):
    txt = open(f'/kaggle/working/log{k}.txt').read()
    print('\n'.join(l for l in txt.split('\n') if 'detector check' in l or 'test done' in l or 'Error' in l or 'error' in l)); print(txt[-1500:])

# %%
DETS = {}
for tag in ('m27', 'm31'):
    if not os.path.exists(f'/kaggle/working/{tag}_oof_f0.pkl'):
        continue
    oof = {}
    for f in range(5):
        oof.update(pickle.load(open(f'/kaggle/working/{tag}_oof_f{f}.pkl', 'rb')))
    test = {f: pickle.load(open(f'/kaggle/working/{tag}_test_f{f}.pkl', 'rb')) for f in range(5)}
    DETS[tag] = dict(oof=oof, test=test)
    sc = np.array([s for v in oof.values() for _, s in v[0]])
    print(tag, 'out-of-fold images', len(oof), 'views', len(next(iter(oof.values()))), 'identity detections', len(sc), 'score quantiles',
          np.round(np.quantile(sc, [.1, .25, .5, .75, .9]), 3).tolist(), flush=True)
pickle.dump(DETS, open('/kaggle/working/detections31.pkl', 'wb'))
print('saved', sorted(os.listdir('/kaggle/working')))
