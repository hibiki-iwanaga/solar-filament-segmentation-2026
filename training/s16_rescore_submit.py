# %% [markdown]
# # Stage 16 (final run): second quality model on the refined outlines, final selection, submission (CPU)
# Own pipeline. Same code as `filament-v46-rescore-v45` (the code of `filament-v36-rescore-v35` with a line patch; both
# are attached and read from their `__notebook__.ipynb`), applied to the outputs of `filament-s14-pool-quality-final`
# and `filament-s15-refine-final`, with these additions for inference without retraining:
# * the three fitted LightGBM models of the selected second-stage score are saved (`stage2_models/`) together with
#   the chosen setting (score, paste-back threshold, keep threshold, probability filter) and checked against the
#   test predictions;
# * the final out-of-fold instances at the chosen setting are saved (`oof_final.pkl`) for the evaluation notebook;
# * the fully patched source is written to `s16_source.py`.
#
# This file is the fully patched source that this run executed (its `s16_source.py`).

# %%
import os, glob, json, time, pickle
from collections import defaultdict
import numpy as np, pandas as pd, cv2
from pycocotools import mask as mu
from multiprocessing import Pool
cv2.setNumThreads(0)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 40)

ROOT = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026', recursive=True)[0]
TRJ = glob.glob(ROOT + '/train/*.json')[0]
TED = ROOT + '/test/test_images'
SELP = glob.glob('/kaggle/input/**/selected_v43.pkl', recursive=True)[0]
d = pickle.load(open(SELP, 'rb'))
RD = os.path.dirname(glob.glob('/kaggle/input/**/info_*.pkl', recursive=True)[0])
VAR = d['winner'] if os.path.exists(f"{RD}/info_{d['winner']}.pkl") else os.path.basename(sorted(glob.glob(RD + '/info_*.pkl'))[0])[5:-4]
SEL = d['sel'][VAR]
INFO = pickle.load(open(f'{RD}/info_{VAR}.pkl', 'rb'))
CANDP = glob.glob('/kaggle/input/**/candidates_v43.pkl', recursive=True)[0]
print('variant', VAR, flush=True)
print(ROOT, SELP, SEL['lam'], RD, flush=True)
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
def annot(sid, s):
    sid = str(sid)
    return sid[:-(len(s) + 1)] if sid.endswith(s) and len(sid) > len(s) + 1 else sid.split('-')[0]
ANN = {s: [annot(sid, s) for sid, _ in GT[s]] for s in stems}
print('example ids', [str(sid) for sid in list(imgs)[:3]], '->', [ANN[s] for s in stems[:3]], flush=True)

# %%
def frame(seed):
    ys, xs = np.nonzero(seed)
    cx, cy = xs.mean(), ys.mean()
    sxx, syy, sxy = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean(), ((xs - cx) * (ys - cy)).mean()
    phi = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    c, sn = np.cos(phi), np.sin(phi)
    u = (xs - cx) * c + (ys - cy) * sn; v = -(xs - cx) * sn + (ys - cy) * c
    um, vm = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    cx, cy = cx + um * c - vm * sn, cy + um * sn + vm * c
    L, Wd = u.max() - u.min() + 1, v.max() - v.min() + 1
    sc = min(CW / (L * 1.2 + 48), CH / (Wd * 2.0 + 48), 2.0)
    M = np.array([[sc * c, sc * sn, 0.], [-sc * sn, sc * c, 0.]])
    M[:, 2] = np.array([CW / 2, CH / 2]) - M[:, :2] @ np.array([cx, cy])
    return M
t0 = time.time()
MS = {'train': defaultdict(list), 'test': defaultdict(list)}      # crop geometry exactly as used by the refiners in v29
for split in MS:
    for i in sorted(INFO[split], key=lambda r: (r['stem'], r['k'])):
        MS[split][i['stem']].append(i['M'])
    for s in (stems if split == 'train' else tests):
        assert len(MS[split][s]) == len(SEL[split].get(s, [])), (split, s)
OFF = {}; n = 0
for s in stems:
    OFF[s] = n; n += len(MS['train'][s])
NTR = n
OFT = {}; n = 0
for s in tests:
    OFT[s] = n; n += len(MS['test'][s])
NTE = n
print('instances', NTR, NTE, f'{time.time() - t0:.0f}s', flush=True)
TAGS = sorted(os.path.basename(p)[len(f'ref_{VAR}_'):-4] for p in glob.glob(f'{RD}/ref_{VAR}_*.npz'))
RO = np.zeros((NTR, CH, CW), np.uint16); RT = np.zeros((NTE, CH, CW), np.uint16)
for tg in TAGS:
    z = np.load(f'{RD}/ref_{VAR}_{tg}.npz')
    assert len(z['ro']) == NTR and len(z['rt']) == NTE, (tg, len(z['ro']), NTR, len(z['rt']), NTE)
    RO += z['ro']; RT += z['rt']
RO = ((RO + len(TAGS) // 2) // len(TAGS)).astype(np.uint8); RT = ((RT + len(TAGS) // 2) // len(TAGS)).astype(np.uint8)
print('refiners', TAGS, flush=True)
print('refined crops loaded', RO.shape, RT.shape, f'{time.time() - t0:.0f}s', flush=True)
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

# %%
CAND = pickle.load(open(CANDP, 'rb'))
CK = [k for k in CAND if k[1] == 'train'][0][0]
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
TD = 0.40; TPS = (0.35, 0.40, 0.45)      # threshold of the descriptors / thresholds of the final passes (v29: 0.35 and 0.45 within 0.001)
def describe(job):
    """Independent (not yet disjoint) refined outline of every instance of one image + descriptors (+ scores)."""
    split, s = job
    cv2.setNumThreads(1)
    sel = SEL[split].get(s, [])
    if not sel:
        return split, s, []
    preds = (RO[OFF[s]:OFF[s] + len(sel)] if split == 'train' else RT[OFT[s]:OFT[s] + len(sel)])
    Ms = MS[split][s]
    cf = {c[0]['counts'].decode() if isinstance(c[0]['counts'], bytes) else c[0]['counts']: c[1] for c in CAND[(CK, split)][s][0]}
    us = np.array([q for _, q in sel]); rk = np.argsort(np.argsort(-us))
    out = []; full = np.zeros((H, W), np.uint8, order='F'); rles = []
    for k, (c, u) in enumerate(sel):
        seed = mu.decode(dict(size=[H, W], counts=c.encode())).astype(bool)
        pb, pv, x0, y0 = paste(preds[k], Ms[k], TD)
        sb = seed[y0:y0 + pb.shape[0], x0:x0 + pb.shape[1]]
        n, lab = cv2.connectedComponents(pb.astype(np.uint8), connectivity=8)
        keep = np.unique(lab[pb & sb]); keep = keep[keep > 0]
        m = np.isin(lab, keep) if len(keep) else np.zeros(pb.shape, bool)
        a_s, a_m = float(seed.sum()), float(m.sum())
        f = dict(cf.get(c, {}))
        f.update(u=u, u_rank=int(rk[k]), n_sel=len(sel), found=int(c in cf), sc=float(np.sqrt(abs(np.linalg.det(Ms[k][:, :2])))),
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
    if split == 'train':
        Sm = np.zeros((len(sel), KOF[s])); Ss = np.zeros((len(sel), KOF[s]))
        srl = [dict(size=[H, W], counts=c.encode()) for c, _ in sel]
        for j, (_, g) in enumerate(GT[s]):
            if not g:
                continue
            if ok:
                iou = np.array(mu.iou([rles[i] for i in ok], g, [0] * len(g))); Sm[ok, j] = np.where(iou > .5, iou, 0).max(1)
            iou = np.array(mu.iou(srl, g, [0] * len(g))); Ss[:, j] = np.where(iou > .5, iou, 0).max(1)
        for k, f in enumerate(out):
            f.update(q=float(Sm[k].mean()), q_seed=float(Ss[k].mean()))
    for k, f in enumerate(out):
        f.update(stem=s, k=k)
    return split, s, out
rows = {'train': [], 'test': []}
with Pool(4) as pool:
    for split, s, out in pool.imap_unordered(describe, [('train', s) for s in stems] + [('test', s) for s in tests], chunksize=4):
        rows[split] += out
DTR = pd.DataFrame(rows['train']).sort_values(['stem', 'k']).reset_index(drop=True)
DTE = pd.DataFrame(rows['test']).sort_values(['stem', 'k']).reset_index(drop=True)
DTR['fold'] = DTR.stem.map(FOLD)
print('refined instances', len(DTR), len(DTE), 'seed features found', round(DTR.found.mean(), 4), round(DTE.found.mean(), 4),
      '| mean q refined', round(DTR.q.mean(), 4), 'mean q seed', round(DTR.q_seed.mean(), 4), f'{time.time() - t0:.0f}s', flush=True)

# %%
import lightgbm as lgb
DROP = ('stem', 'k', 'q', 'q_seed', 'fold', 'found')
FE_ALL = [c for c in DTR.columns if c not in DROP and c in DTE.columns]
FE_R = [c for c in FE_ALL if c.startswith('r_') or c in ('u', 'u_rank', 'n_sel', 'sc', 'crop_max', 'crop_mean')]
PRMS = {'small': dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=4),
        'mid': dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=4)}
MODELS = {'r_small': (FE_R, 'small'), 'all_small': (FE_ALL, 'small'), 'all_mid': (FE_ALL, 'mid')}
for name, (fe, prm) in MODELS.items():
    DTR[name] = 0.
    for sd in range(3):
        for f in range(5):
            tr, va = DTR.fold.values != f, DTR.fold.values == f
            DTR.loc[va, name] += lgb.LGBMRegressor(**PRMS[prm], random_state=sd).fit(DTR.loc[tr, fe], DTR.loc[tr, 'q']).predict(DTR.loc[va, fe]) / 3
    print(name, 'features', len(fe), 'corr with q', round(float(np.corrcoef(DTR[name], DTR.q)[0, 1]), 4), '| corr of u', round(float(np.corrcoef(DTR.u, DTR.q)[0, 1]), 4),
          '| rmse', round(float(np.sqrt(((DTR[name] - DTR.q) ** 2).mean())), 4), 'u', round(float(np.sqrt(((DTR.u - DTR.q) ** 2).mean())), 4), flush=True)
m_ = lgb.LGBMRegressor(**PRMS['small']).fit(DTR[FE_ALL], DTR.q)
print(pd.Series(m_.feature_importances_, FE_ALL).sort_values(ascending=False).head(25).to_dict(), flush=True)
SCORES = {'u': DTR.u.values, **{n: DTR[n].values for n in MODELS}}
SCORES['mix'] = 0.5 * DTR.u.values + 0.5 * DTR['all_small'].values
PMV = DTR.r_pm.fillna(0).values
QS = {}

def image_pass(job):
    name, tp, s = job
    cv2.setNumThreads(1)
    sel = SEL['train'].get(s, [])
    if not sel:
        return job, None
    q = QS[name][OFF[s]:OFF[s] + len(sel)]
    res = refine_seq([(c, float(qq)) for (c, _), qq in zip(sel, q)], RO[OFF[s]:OFF[s] + len(sel)], MS['train'][s], tp)
    kept = np.array([r is not None for r, _ in res]); pm = np.array([p for _, p in res])
    miou = np.zeros((len(sel), KOF[s]))
    rles = [r for r, _ in res if r is not None]; ki = np.where(kept)[0]
    for a, (_, g) in enumerate(GT[s]):
        if g and rles:
            iou = np.array(mu.iou(rles, g, [0] * len(g)))
            miou[ki, a] = np.where(iou > .5, iou, 0).max(1)
    return job, dict(u=np.asarray(q, float), kept=kept, pm=pm, miou=miou)
QS = SCORES
cv2.setNumThreads(0)
PRES = defaultdict(dict)
with Pool(4) as pool:
    for (name, tp, s), r in pool.imap_unordered(image_pass, [(n, tp, s) for n in SCORES for tp in TPS for s in stems], chunksize=8):
        if r is not None:
            PRES[(name, tp)][s] = r
RECS = [(s, i) for s in stems for i in range(KOF[s])]
K2R = [r for r in RECS if KOF[r[0]] >= 2]
def score(PR, lam, pmin=0., recs=None):
    num = 0.; tp = npk = ngt = 0
    for s, i in (RECS if recs is None else recs):
        ngt += len(GT[s][i][1])
        r = PR.get(s)
        if r is None:
            continue
        keep = r['kept'] & (r['u'] > lam) & (r['pm'] >= pmin)
        npk += int(keep.sum()); num += r['miou'][keep, i].sum(); tp += int((r['miou'][keep, i] > 0).sum())
    return dict(pq=num / max(tp + .5 * (npk - tp) + .5 * (ngt - tp), 1e-9), tp=tp, fp=npk - tp, fn=ngt - tp, sq=num / max(tp, 1),
                per_img=npk / max(len(RECS if recs is None else recs), 1))
LAMS = [round(x, 2) for x in np.arange(0.10, 0.44, 0.02)]
R = {}
for name in SCORES:
    for tp in TPS:
        for pmin in (0., 0.5):
            for lam in LAMS:
                R[(name, tp, lam, pmin)] = score(PRES[(name, tp)], lam, pmin)
            print(f'{name:10s} t {tp} pmin {pmin}', ' '.join(f"{lam}:{R[(name, tp, lam, pmin)]['pq']:.4f}" for lam in LAMS[2::2]), flush=True)
    b = max((k for k in R if k[0] == name), key=lambda k: R[k]['pq'])
    print('  best', b, {a: round(float(v), 4) for a, v in R[b].items()}, 'K2+', round(score(PRES[(name, b[1])], b[2], b[3], K2R)['pq'], 4), flush=True)
print('baseline (u, t 0.35, lam 0.26, prob >= 0.5):', {a: round(float(v), 4) for a, v in R[('u', 0.35, 0.26, 0.5)].items()}, flush=True)
BEST = max((k for k in R if k[0] != 'u'), key=lambda k: R[k]['pq'])
print('BEST', BEST, {a: round(float(v), 4) for a, v in R[BEST].items()}, flush=True)

# %%
name, tp, lam, pmin = BEST
if name == 'mix':
    fe, prm = MODELS['all_small']
else:
    fe, prm = MODELS[name]
M2 = [lgb.LGBMRegressor(**PRMS[prm], random_state=sd).fit(DTR[fe], DTR.q) for sd in range(3)]
qt = np.mean([m.predict(DTE[fe]) for m in M2], 0)
os.makedirs('/kaggle/working/stage2_models', exist_ok=True)
for sd, m in enumerate(M2):
    m.booster_.save_model(f'/kaggle/working/stage2_models/seed{sd}.txt')
_q2 = np.mean([lgb.Booster(model_file=f'/kaggle/working/stage2_models/seed{sd}.txt').predict(DTE[fe].values.astype(np.float64)) for sd in range(3)], 0)
print('saved boosters reproduce the test predictions: max abs difference', float(np.abs(_q2 - qt).max()), flush=True)
json.dump(dict(score=name, features=list(fe), threshold=float(tp), lam=float(lam), pmin=float(pmin), descriptor_threshold=float(TD), variant=VAR,
               pq_oof=float(R[BEST]['pq'])), open('/kaggle/working/stage2_models/model.json', 'w'), indent=1)
if name == 'mix':
    qt = 0.5 * DTE.u.values + 0.5 * qt
print('test: mean second-stage quality', round(float(qt.mean()), 4), 'oof', round(float(SCORES[name].mean()), 4),
      '| share above lam: test', round(float((qt > lam).mean()), 4), 'oof', round(float((SCORES[name] > lam).mean()), 4), flush=True)
def sub_rows(q_all, tp, lam, pmin):
    rows = []
    for s in tests:
        sel = SEL['test'].get(s, [])
        if not sel:
            continue
        q = q_all[OFT[s]:OFT[s] + len(sel)]
        sel2 = [(c, float(qq)) for (c, _), qq in zip(sel, q)]
        preds = RT[OFT[s]:OFT[s] + len(sel)]; Ms = MS['test'][s]
        use = q > lam
        if pmin > 0:
            pm = np.array([p for _, p in refine_seq(sel2, preds, Ms, tp, 0, use=use)])
            use &= pm >= pmin
        res = refine_seq(sel2, preds, Ms, tp, 0, use=use)
        assert len(res) == len(sel2)
        kept = [(r, float(qq), float(u0)) for (r, _), qq, (_, u0) in zip(res, q, sel) if r is not None]
        rows += [(f'{s}_{k + 1}', r['counts'].decode()) for k, (r, _, _) in enumerate(kept)]
        QROWS.extend((f'{s}_{k + 1}', qq, u0) for k, (_, qq, u0) in enumerate(kept))
    return pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
QROWS = []
assert list(DTE.stem.values) == [s for s in tests for _ in SEL['test'].get(s, [])]
sub = sub_rows(qt, tp, lam, pmin)
sub.to_csv('/kaggle/working/submission.csv', index=False)
pd.DataFrame(QROWS, columns=['filament_id', 'q', 'u']).to_csv('/kaggle/working/submission_quality.csv', index=False)      # predicted quality of every row (for the consensus notebook)
for s in tests[:60]:
    rs = [dict(size=[H, W], counts=c.encode()) for c in sub[sub.filament_id.str.startswith(s + '_')].segmentation_rle]
    if len(rs) > 1:
        ov = np.array(mu.iou(rs, rs, [0] * len(rs))); np.fill_diagonal(ov, 0)
        assert ov.max() == 0
print('submission', BEST, sub.shape, f'{len(sub) / len(tests):.2f} per image', flush=True)
DTR.drop(columns=['stem']).assign(stem=DTR.stem).to_pickle('/kaggle/working/refined_table_oof.pkl'); DTE.to_pickle('/kaggle/working/refined_table_test.pkl')
json.dump({'_'.join(map(str, k)): v for k, v in R.items()}, open('/kaggle/working/summary_v46.json', 'w'), indent=1, default=float)
OOFF = {}
cv2.setNumThreads(1)
for s in stems:                                   # final out-of-fold instances at the chosen setting (same code path as the test)
    sel = SEL['train'].get(s, [])
    if not sel:
        OOFF[s] = []; continue
    q = SCORES[name][OFF[s]:OFF[s] + len(sel)]
    sel2 = [(c, float(qq)) for (c, _), qq in zip(sel, q)]
    preds = RO[OFF[s]:OFF[s] + len(sel)]; Ms = MS['train'][s]
    use = q > lam
    if pmin > 0:
        pm = np.array([p for _, p in refine_seq(sel2, preds, Ms, tp, 0, use=use)])
        use &= pm >= pmin
    OOFF[s] = [(r['counts'].decode(), float(qq)) for (r, _), qq in zip(refine_seq(sel2, preds, Ms, tp, 0, use=use), q) if r is not None]
pickle.dump(OOFF, open('/kaggle/working/oof_final.pkl', 'wb'))
_num = _tp = _np = _ng = 0
for s in stems:
    rl = [dict(size=[H, W], counts=c.encode()) for c, _ in OOFF[s]]
    for _, g in GT[s]:
        _ng += len(g); _np += len(rl)
        if g and rl:
            iou = np.array(mu.iou(rl, g, [0] * len(g))); m = np.where(iou > .5, iou, 0).max(1)
            _num += m.sum(); _tp += int((m > 0).sum())
print('out-of-fold PQ of the saved final instances', round(float(_num / (_tp + .5 * (_np - _tp) + .5 * (_ng - _tp))), 4),
      'per image', round(float(np.mean([len(v) for v in OOFF.values()])), 2), flush=True)
print('done', f'{time.time() - t0:.0f}s', flush=True)

# %%
import hashlib, lightgbm
print('md5 submission.csv', hashlib.md5(open('/kaggle/working/submission.csv', 'rb').read()).hexdigest(), 'rows', len(sub), 'lightgbm', lightgbm.__version__)
for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True):
    a = pd.read_csv(p)
    same = len(set(a.segmentation_rle) & set(sub.segmentation_rle))
    print('earlier submission', p, a.shape, 'identical masks', same, flush=True)
print(sorted(os.listdir('/kaggle/working')), sorted(os.listdir('/kaggle/working/stage2_models')))

# %% [markdown]
# ## Out-of-fold evaluation of the final instances (for the report)
# Same matching as the competition metric and the organizers' self-evaluation notebook: every annotator record of an
# image is scored separately against all predictions of the image; IoU > 0.5 is a match. Also: IoU and Dice of the
# matched pairs, best IoU of every annotated filament, pixel Dice of the union masks per record, and the overlap
# counts of the self-evaluation notebook (one-to-many: an annotated filament touched by several predictions;
# many-to-one: a prediction touching several annotated filaments; IoU > 0).

# %%
ev_gt, ev_pr, ev_rec = [], [], []
for s in stems:
    rl = [dict(size=[H, W], counts=c.encode()) for c, _ in OOFF[s]]
    Pm = np.zeros((H, W), bool)
    for r in rl:
        Pm |= mu.decode(r).astype(bool)
    for sid, g in GT[s]:
        if not g:
            continue
        iou = np.array(mu.iou(rl, g, [0] * len(g))).reshape(len(rl), len(g)) if rl else np.zeros((0, len(g)))
        for j in range(len(g)):
            b = float(iou[:, j].max()) if len(rl) else 0.
            ev_gt.append(dict(stem=s, rec=str(sid), K=KOF[s], best_iou=b, n_touch=int((iou[:, j] > 0).sum()), match=b > .5))
        for i in range(len(rl)):
            ev_pr.append(dict(stem=s, rec=str(sid), K=KOF[s], best_iou=float(iou[i].max()), n_touch=int((iou[i] > 0).sum()), match=bool(iou[i].max() > .5)))
        Gm = np.zeros((H, W), bool)
        for r in g:
            Gm |= mu.decode(r).astype(bool)
        inter = float((Pm & Gm).sum())
        ev_rec.append(dict(stem=s, rec=str(sid), K=KOF[s], dice=2 * inter / max(float(Pm.sum() + Gm.sum()), 1.), iou=inter / max(float((Pm | Gm).sum()), 1.)))
EG, EP, ER = pd.DataFrame(ev_gt), pd.DataFrame(ev_pr), pd.DataFrame(ev_rec)
def summary(eg, ep):
    tp = int(eg.match.sum()); fn = len(eg) - tp; fp = int((~ep.match).sum())
    ti = eg.best_iou[eg.match]
    return dict(PQ=float(ti.sum() / (tp + .5 * fp + .5 * fn)), SQ=float(ti.mean()), RQ=float(tp / (tp + .5 * fp + .5 * fn)), TP=tp, FP=fp, FN=fn,
                matched_iou_q=[round(float(x), 4) for x in np.quantile(ti, [.1, .25, .5, .75, .9])],
                matched_dice_mean=float((2 * ti / (1 + ti)).mean()), one_to_many=float((eg.n_touch > 1).mean()), many_to_one=float((ep.n_touch > 1).mean()))
EV = {'all': summary(EG, EP), 'K1': summary(EG[EG.K == 1], EP[EP.K == 1]), 'K2+': summary(EG[EG.K >= 2], EP[EP.K >= 2])}
EV['pixel_dice_per_record'] = dict(mean=float(ER.dice.mean()), median=float(ER.dice.median()), q=[round(float(x), 4) for x in np.quantile(ER.dice, [.1, .25, .5, .75, .9])])
EV['pixel_iou_per_record'] = dict(mean=float(ER.iou.mean()), median=float(ER.iou.median()))
EV['best_iou_hist'] = np.histogram(EG.best_iou, bins=20, range=(0, 1))[0].tolist()
EV['instances_per_image'] = float(np.mean([len(v) for v in OOFF.values()]))
EV['matched_iou_hist'] = np.histogram(EG.best_iou[EG.match], bins=10, range=(0.5, 1))[0].tolist()
EV['record_dice_hist'] = np.histogram(ER.dice, bins=20, range=(0, 1))[0].tolist()
# bootstrap over images: 95 % interval of the pooled PQ (error bar for the report)
per = {}
for eg in EG.itertuples():
    a = per.setdefault(eg.stem, [0., 0, 0, 0]); a[0] += eg.best_iou if eg.match else 0.; a[1] += int(eg.match); a[3] += int(not eg.match)
for ep in EP.itertuples():
    a = per.setdefault(ep.stem, [0., 0, 0, 0]); a[2] += int(not ep.match)
A_ = np.array(list(per.values()))
rng_ = np.random.RandomState(0); bs = []
for _ in range(2000):
    t_ = A_[rng_.randint(0, len(A_), len(A_))].sum(0); bs.append(t_[0] / (t_[1] + .5 * t_[2] + .5 * t_[3]))
EV['PQ_bootstrap_95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
EV['PQ_by_fold'] = {}
for f in range(5):
    m_ = [FOLD[s] == f for s in per]
    t_ = A_[np.array(m_)].sum(0); EV['PQ_by_fold'][f] = float(t_[0] / (t_[1] + .5 * t_[2] + .5 * t_[3]))
json.dump(EV, open('/kaggle/working/eval_oof.json', 'w'), indent=1)
EG.to_csv('/kaggle/working/eval_oof_annotations.csv', index=False); EP.to_csv('/kaggle/working/eval_oof_predictions.csv', index=False)
ER.to_csv('/kaggle/working/eval_oof_records.csv', index=False)
print(json.dumps(EV, indent=1), flush=True)
