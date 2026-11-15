"""End-to-end inference: test images -> submission file, using only the trained weights in `weights/`.

    python -m solfil.pipeline --images <folder of test images> --weights weights --cache /tmp/solfil --out submission.csv

Steps (stage numbers refer to `training/`):
  1. preprocessing: image, limb-flattened image, 1024 px copies                            (memory-mapped cache)
  2. networks: dense maps B, C, D (stages 02-04), simulated-annotator maps E (05), Mask R-CNN detections m31, m34
     (12, 13); on two GPUs the networks run in two worker processes
  3. stage 14: calibrated ensemble map -> candidate pool -> descriptors -> first quality model -> greedy selection
  4. stage 15: oriented crops of the selected instances -> three refiners (stages 09-11), mean probability
  5. stage 16: descriptors of the refined outlines -> second quality model -> final pixel-disjoint instances
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd

from . import preprocess as pp

cv2.setNumThreads(0)      # no OpenCV worker threads in the parent: forked workers would deadlock on them
FOLDS = range(int(os.environ.get('SOLFIL_FOLDS', '5')))      # 5 fold models per network (fewer only for smoke tests)
WEIGHTS = {'B': 'dense_b_effb4', 'C': 'dense_c_r34_2048', 'D': 'dense_d_unetpp_spine', 'E': 'sim_annotators',
           'm31': 'detector_1024', 'm34': 'detector_2048',
           'v17': 'refiner_unetpp_r34', 'v18': 'refiner_unet_resnext50', 'v19': 'refiner_unet_effb4'}
GPU_PLAN = (('C', 'B', 'D', 'm31'), ('m34', 'E'))       # two workers of about the same length on two GPUs


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def wpath(weights, key, f):
    return os.path.join(weights, WEIGHTS[key], f'fold{f}.pt')


# ---------------------------------------------------------------- 1. preprocessing

def _prep_one(path):
    cv2.setNumThreads(1)
    im = pp.read_gray(path)
    if im.shape != (pp.H, pp.W):
        im = cv2.resize(im, (pp.W, pp.H), interpolation=cv2.INTER_AREA)
    fl = pp.flatten(im)
    return im, fl, pp.small(im), pp.small(fl)


def prepare(image_dir, cache, workers=4):
    os.makedirs(cache, exist_ok=True)
    stems, paths = pp.list_images(image_dir)
    n = len(stems)
    img = np.lib.format.open_memmap(f'{cache}/img2k.npy', 'w+', np.uint8, (n, pp.H, pp.W))
    flat = np.lib.format.open_memmap(f'{cache}/flat2k.npy', 'w+', np.uint8, (n, pp.H, pp.W))
    i1k = np.lib.format.open_memmap(f'{cache}/img1k.npy', 'w+', np.uint8, (n, 2, pp.S1, pp.S1))
    with get_context('fork').Pool(workers) as pool:
        for i, (im, fl, a, b) in enumerate(pool.imap(_prep_one, [paths[s] for s in stems], chunksize=2)):
            img[i], flat[i], i1k[i, 0], i1k[i, 1] = im, fl, a, b
    img.flush(); flat.flush(); i1k.flush()
    json.dump(dict(stems=stems, paths=[paths[s] for s in stems]), open(f'{cache}/meta.json', 'w'))
    log('prepared', n, 'images')
    return stems


# ---------------------------------------------------------------- 2. networks

def gpu_tasks(tasks, cache, weights, device=None):
    """Run the networks of `tasks` on one device and write their test outputs to the cache."""
    import torch
    from . import nets
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    stems = json.load(open(f'{cache}/meta.json'))['stems']
    n = len(stems)
    img = np.load(f'{cache}/img2k.npy', mmap_mode='r'); flat = np.load(f'{cache}/flat2k.npy', mmap_mode='r')
    i1k = np.load(f'{cache}/img1k.npy', mmap_mode='r')
    for task in tasks:
        t0 = time.time()
        if task == 'B':
            med = [pp.disk_median(i1k[i, 0]) for i in range(n)]
            per = []
            for f in FOLDS:
                m = nets.build_dense('B'); m.load_state_dict(nets.load_state(wpath(weights, 'B', f))); m.to(dev).eval()
                per.append(np.stack([nets.predict_b(m, np.array(i1k[i, 0]), med[i], dev) for i in range(n)]))
                np.save(f'{cache}/B_f{f}.npy', per[-1]); del m
            np.save(f'{cache}/B_mean.npy', np.mean([p.astype(np.float32) for p in per], 0).astype(np.uint8))
        elif task == 'C':
            med = [pp.disk_median(img[i]) for i in range(n)]
            acc = np.zeros((n, pp.H, pp.W), np.uint16)
            for f in FOLDS:
                m = nets.build_dense('C'); m.load_state_dict(nets.load_state(wpath(weights, 'C', f))); m.to(dev).eval()
                o1 = np.zeros((n, pp.S1, pp.S1), np.uint8)
                for i in range(n):
                    a2, o1[i] = nets.predict_c(m, np.array(img[i]), np.array(flat[i]), med[i], dev)
                    acc[i] += a2
                np.save(f'{cache}/C_f{f}.npy', o1); del m
            np.save(f'{cache}/C_mean2k.npy', ((acc + 2) // 5).astype(np.uint8))
        elif task == 'D':
            med = [pp.disk_median(i1k[i, 0]) for i in range(n)]
            per = []
            for f in FOLDS:
                m = nets.build_dense('D'); m.load_state_dict(nets.load_state(wpath(weights, 'D', f))); m.to(dev).eval()
                per.append(np.stack([nets.predict_d(m, np.array(i1k[i]), med[i], dev) for i in range(n)]))
                np.save(f'{cache}/D_f{f}.npy', per[-1][:, 0]); del m
            np.save(f'{cache}/D_mean.npy', np.mean([p[:, 0].astype(np.float32) for p in per], 0).round().astype(np.uint8))
            np.save(f'{cache}/Dsp_mean.npy', np.mean([p[:, 1].astype(np.float32) for p in per], 0).round().astype(np.uint8))
        elif task == 'E':
            info = json.load(open(os.path.join(weights, WEIGHTS['E'], 'annotators.json')))
            A = len(info['anns'])
            models = []
            for f in FOLDS:
                m = nets.CondUnet(A); m.load_state_dict(nets.load_state(wpath(weights, 'E', f))); models.append(m.to(dev).eval())
            pa8 = np.lib.format.open_memmap(f'{cache}/E_pa8.npy', 'w+', np.uint8, (n, A, pp.S1, pp.S1))
            for i in range(n):
                med = pp.disk_median(i1k[i, 0])
                pa = 0
                for m in models:
                    pa = pa + nets.predict_annotators(m, np.array(i1k[i]), med, A, dev)
                pa8[i] = (pa / len(models) * 255).round().byte().cpu().numpy()
            pa8.flush(); del models
        elif task in ('m31', 'm34'):
            native = task == 'm34'
            det = {}
            for f in FOLDS:
                m = nets.build_detector(1024); m.load_state_dict(nets.load_state(wpath(weights, task, f))); m.to(dev).eval()
                det[f] = {}
                for i, s in enumerate(stems):
                    x2 = np.stack([img[i], flat[i]]) if native else np.array(i1k[i])
                    det[f][s] = [nets.detect(m, x2, v, native, dev) for v in nets.VIEWS]
                del m
            pickle.dump(det, open(f'{cache}/{task}.pkl', 'wb'))
        else:
            raise ValueError(task)
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        log('network', task, 'done', f'{time.time() - t0:.0f}s')


def run_networks(cache, weights):
    """All networks; with two GPUs in two worker processes."""
    import torch
    ng = torch.cuda.device_count()
    if ng >= 2:
        procs = []
        for g, tasks in enumerate(GPU_PLAN):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            logf = open(f'{cache}/gpu{g}.log', 'w')
            procs.append((subprocess.Popen([sys.executable, '-m', 'solfil.pipeline', 'gpu', '--tasks', ','.join(tasks), '--cache', cache,
                                            '--weights', weights], env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
        while any(p.poll() is None for p, _ in procs):
            time.sleep(60)
            for g in range(len(procs)):
                ls = open(f'{cache}/gpu{g}.log').read().strip().split('\n')
                log(f'gpu{g}:', ls[-1][:160])
        for g, (p, _) in enumerate(procs):
            if p.returncode != 0:
                raise RuntimeError(f'GPU worker {g} failed:\n' + open(f'{cache}/gpu{g}.log').read()[-3000:])
    else:
        gpu_tasks([t for ts in GPU_PLAN for t in ts], cache, weights)


# ---------------------------------------------------------------- 3. stage 14

def sim_levels(lut, pa8):
    """Quantile-matched per-annotator maps [A, 1024, 1024] -> levels 0..3 (probability >= 0.3 / 0.5 / 0.7)."""
    q = lut[np.asarray(pa8)]
    return (q >= 77).astype(np.uint8) + (q >= 128) + (q >= 179)


_G = {}


def _init_stage1(cache, team):
    cv2.setNumThreads(1)
    _G.update(stems=json.load(open(f'{cache}/meta.json'))['stems'], img=np.load(f'{cache}/img2k.npy', mmap_mode='r'),
              prob=np.load(f'{cache}/prob2k.npy', mmap_mode='r'), std=np.load(f'{cache}/std1k.npy', mmap_mode='r'),
              spine=np.load(f'{cache}/spine1k.npy', mmap_mode='r'), cache=cache, team=np.array(team),
              pa8=np.load(f'{cache}/E_pa8.npy', mmap_mode='r'), elut=np.load(f'{cache}/E_lut.npy'),
              det={t: pickle.load(open(f'{cache}/{t}.pkl', 'rb')) for t in ('m31', 'm34')})


def _stage1_job(i):
    from .candidates import candidates, overlaps, sim_features, det_features
    s = _G['stems'][i]
    cs = candidates(np.array(_G['prob'][i]), np.array(_G['std'][i]), np.array(_G['spine'][i]), np.array(_G['img'][i]))
    ov = overlaps(cs)
    sim_features(cs, sim_levels(_G['elut'], _G['pa8'][i]), _G['team'])
    det_features(cs, {t: {f: d[f].get(s, [[]] * 4) for f in d} for t, d in _G['det'].items()})
    return s, cs, ov


def stage1(cache, weights, workers=4):
    from .candidates import greedy
    import lightgbm as lgb
    cal = json.load(open(os.path.join(weights, 'stage1', 'calibration.json')))
    cfg = json.load(open(os.path.join(weights, 'stage1', 'model.json')))
    stems = json.load(open(f'{cache}/meta.json'))['stems']
    n = len(stems)
    t0 = time.time()
    # calibrated ensemble map B + C + D at 2048 px, member disagreement at 1024 px, calibrated spine map
    B, C2, D = np.load(f'{cache}/B_mean.npy'), np.load(f'{cache}/C_mean2k.npy', mmap_mode='r'), np.load(f'{cache}/D_mean.npy')
    tm = np.lib.format.open_memmap(f'{cache}/ens2k.npy', 'w+', np.uint8, (n, pp.H, pp.W))
    for i in range(n):
        tm[i] = pp.mean_u8([pp.up(B[i]), np.array(C2[i]), pp.up(D[i])])
    lut = pp.lut_from_hist(cal['h_BCD'], sum(np.bincount(tm[i].ravel(), minlength=256) for i in range(n)))
    prob = np.lib.format.open_memmap(f'{cache}/prob2k.npy', 'w+', np.uint8, (n, pp.H, pp.W))
    for i in range(n):
        prob[i] = lut[tm[i]]
    prob.flush(); del tm
    acc = np.zeros((n, pp.S1, pp.S1), np.float32)
    for f in FOLDS:
        Bf, Cf, Df = np.load(f'{cache}/B_f{f}.npy'), np.load(f'{cache}/C_f{f}.npy'), np.load(f'{cache}/D_f{f}.npy')
        for i in range(n):
            acc[i] += pp.spread([Bf[i], Cf[i], Df[i]])
    np.save(f'{cache}/std1k.npy', (acc / 5).round().astype(np.uint8)); del acc
    np.save(f'{cache}/spine1k.npy', pp.quantile_match(cal['h_spine'], np.load(f'{cache}/Dsp_mean.npy')))
    # simulated annotators: quantile-matched per-annotator maps -> levels 0..3
    pa8 = np.load(f'{cache}/E_pa8.npy', mmap_mode='r')
    np.save(f'{cache}/E_lut.npy', pp.lut_from_hist(cal['h_sim'], sum(np.bincount(np.asarray(pa8[i]).ravel(), minlength=256) for i in range(n))))
    log('maps calibrated', f'{time.time() - t0:.0f}s')
    # candidates and descriptors
    team = [a - 1 for a in json.load(open(os.path.join(weights, WEIGHTS['E'], 'annotators.json')))['team']]
    res = {}
    with get_context('fork').Pool(workers, initializer=_init_stage1, initargs=(cache, team)) as pool:
        for k, (s, cs, ov) in enumerate(pool.imap_unordered(_stage1_job, range(n), chunksize=1)):
            res[s] = (cs, ov)
            if k % 30 == 0:
                log('candidates', k, n, f'{time.time() - t0:.0f}s')
    res = {s: res[s] for s in stems}                      # fixed image order
    # first quality model and greedy selection down to the low threshold
    fe = cfg['features']
    bst = {nm: lgb.Booster(model_file=os.path.join(weights, 'stage1', f'{nm}.txt')) for nm in (('reg', 'hit', 'iou') if cfg['hurdle'] else ('reg',))}
    sel, cands = {}, {}
    for s, (cs, ov) in res.items():
        cands[s] = {c[0]['counts'].decode(): {k: v for k, v in c[1].items()} for c in cs}
        if not cs:
            sel[s] = []
            continue
        X = pd.DataFrame([c[1] for c in cs]).reindex(columns=fe).values.astype(np.float32)
        pt = bst['reg'].predict(X, num_threads=1)
        if cfg['hurdle']:
            pt = 0.5 * pt + 0.5 * np.clip(bst['hit'].predict(X, num_threads=1), 0, 1) * np.clip(bst['iou'].predict(X, num_threads=1), 0, 1)
        tk = greedy(pt, ov, cfg['lam_low'])
        sel[s] = [(cs[j][0]['counts'].decode(), float(pt[j])) for j in np.where(tk)[0]]
    pickle.dump(dict(sel=sel, cands=cands), open(f'{cache}/stage1.pkl', 'wb'))
    log('stage 1: candidates per image', round(float(np.mean([len(v[0]) for v in res.values()])), 1), 'selected per image (low threshold)',
        round(float(np.mean([len(v) for v in sel.values()])), 2), f'above lam_best {cfg["lam_best"]}:',
        round(float(np.mean([sum(q > cfg['lam_best'] for _, q in v) for v in sel.values()])), 2), f'{time.time() - t0:.0f}s')
    return sel, cands


# ---------------------------------------------------------------- 4.-5. stages 15 and 16

def _crop_job(args):
    from .refine import build_crops
    i, sel = args
    cv2.setNumThreads(1)
    return build_crops(sel, np.array(_G['img'][i]), np.array(_G['prob'][i]), np.array(_G['spine'][i]))


def _describe_job(args):
    from .refine import describe_refined
    sel, preds, Ms, cf, td = args
    cv2.setNumThreads(1)
    return describe_refined(sel, preds, Ms, cf, td)


def _final_job(args):
    from .refine import final_instances
    cv2.setNumThreads(1)
    return final_instances(*args)


def stage2(cache, weights, sel, cands, out_csv, workers=4):
    import torch
    import lightgbm as lgb
    from . import nets
    stems = json.load(open(f'{cache}/meta.json'))['stems']
    t0 = time.time()
    _G.update(img=np.load(f'{cache}/img2k.npy', mmap_mode='r'), prob=np.load(f'{cache}/prob2k.npy', mmap_mode='r'),
              spine=np.load(f'{cache}/spine1k.npy', mmap_mode='r'))
    with get_context('fork').Pool(workers) as pool:
        built = pool.map(_crop_job, [(i, sel[s]) for i, s in enumerate(stems)], chunksize=2)
    crops = np.concatenate([b[0] for b in built]); MS = {s: b[1] for s, b in zip(stems, built)}
    off = np.cumsum([0] + [len(sel[s]) for s in stems])
    log('crops', crops.shape, f'{time.time() - t0:.0f}s')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc = np.zeros(crops.shape[:1] + crops.shape[2:], np.uint16)
    for tag in ('v17', 'v18', 'v19'):
        rt = np.zeros_like(acc)
        for f in FOLDS:
            m = nets.build_refiner(tag); m.load_state_dict(nets.load_state(wpath(weights, tag, f))); m.to(dev)
            rt += nets.predict_refiner(m, crops, dev)
            del m
        acc += ((rt + 2) // 5).astype(np.uint8)
        log('refiner', tag, 'done', f'{time.time() - t0:.0f}s')
    RT = ((acc + 1) // 3).astype(np.uint8)
    np.save(f'{cache}/refined.npy', RT)
    # second quality model on the refined outlines
    cfg = json.load(open(os.path.join(weights, 'stage2', 'model.json')))
    jobs = [(sel[s], RT[off[i]:off[i + 1]], MS[s], cands[s], cfg['descriptor_threshold']) for i, s in enumerate(stems)]
    with get_context('fork').Pool(workers) as pool:
        desc = pool.map(_describe_job, jobs, chunksize=2)
    rows = [dict(f, stem=s, k=k) for s, d in zip(stems, desc) for k, f in enumerate(d)]
    DTE = pd.DataFrame(rows)
    X = DTE.reindex(columns=cfg['features']).values.astype(np.float64)
    q = np.mean([lgb.Booster(model_file=os.path.join(weights, 'stage2', f'seed{sd}.txt')).predict(X, num_threads=1) for sd in range(3)], 0)
    if cfg['score'] == 'mix':
        q = 0.5 * DTE.u.values + 0.5 * q
    jobs = [(sel[s], q[off[i]:off[i + 1]], RT[off[i]:off[i + 1]], MS[s], cfg['threshold'], cfg['lam'], cfg['pmin']) for i, s in enumerate(stems)]
    with get_context('fork').Pool(workers) as pool:
        final = pool.map(_final_job, jobs, chunksize=2)
    rows, qrows = [], []
    for s, inst in zip(stems, final):
        for k, (c, qq) in enumerate(inst):
            rows.append((f'{s}_{k + 1}', c)); qrows.append((f'{s}_{k + 1}', qq))
    sub = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    sub.to_csv(out_csv, index=False)
    pd.DataFrame(qrows, columns=['filament_id', 'q']).to_csv(os.path.splitext(out_csv)[0] + '_quality.csv', index=False)
    DTE.to_pickle(f'{cache}/refined_table.pkl')
    log('submission', out_csv, sub.shape, f'{len(sub) / len(stems):.2f} instances per image', f'{time.time() - t0:.0f}s')
    return sub


def run(images, weights, cache, out_csv, workers=4):
    t0 = time.time()
    prepare(images, cache, workers)
    run_networks(cache, weights)
    sel, cands = stage1(cache, weights, workers)
    sub = stage2(cache, weights, sel, cands, out_csv, workers)
    log('total', f'{time.time() - t0:.0f}s')
    return sub


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', nargs='?', default='run', choices=['run', 'gpu'])
    ap.add_argument('--images', help='folder of test images')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--cache', default='/tmp/solfil')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--tasks', default='', help='(gpu) comma-separated networks to run')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == 'gpu':
        gpu_tasks(a.tasks.split(','), a.cache, a.weights)
    else:
        run(a.images, a.weights, a.cache, a.out, a.workers)


if __name__ == '__main__':
    main()
