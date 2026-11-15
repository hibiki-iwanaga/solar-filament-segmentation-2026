"""Candidate pool and candidate descriptors (stage 14).

A candidate is a connected region of the calibrated ensemble probability map (members B, C, D) at one of five
thresholds, optionally with small gaps closed (11 or 25 px disk) or with nearby pieces grouped (41 or 61 px):
5 thresholds x (3 closing + 2 grouping variants) = 25 settings, about 157 candidates per image.
Every candidate is described by
  * map statistics (probability, member disagreement, spine map of member D),
  * shape (area, solidity, compactness, elongation, width, fragments, holes) and image contrast,
  * nesting with the other candidates (how many lie inside it / contain it),
  * agreement with 40 simulated annotators (member E): how many of them would draw an outline that matches it,
  * agreement with the Mask R-CNN instances (2 detectors x 4 views, averaged over the five fold models),
  * solar-physics descriptors (heliographic position, tilt, fragment linkage): computed, but not used by the
    selected quality model (they did not improve the out-of-fold score, see the report).
"""
import numpy as np
import cv2
from pycocotools import mask as mu

H = W = 2048
TS = (0.2, 0.3, 0.4, 0.5, 0.6)
TS_FINE = (0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.8)
SETTINGS = [(t, c, c) for t in TS for c in (0, 11, 25)] + [(t, 11, g) for t in TS for g in (41, 61)]
KER = {c: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)) for c in (11, 25)}
KERG = {g: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g // 4 + 1, g // 4 + 1)) for g in (41, 61)}
PHYS = ['ph_lat', 'ph_abslat', 'ph_lon', 'ph_abslon', 'ph_mu', 'ph_r', 'ph_tilt', 'ph_tilt_h', 'ph_abstilt', 'ph_rad_ang', 'ph_llen', 'ph_larea',
        'lk_n', 'lk_gap_max', 'lk_gap_mean', 'lk_gap_rel', 'lk_gap_dep', 'lk_second', 'lk_align', 'lk_tiltdiff', 'lk_elong1', 'lk_fit', 'lk_fit_rel',
        'dark_frac', 'ring_bright', 'ring_p5', 'img_sharp', 'img_std', 'img_R', 'img_limb', 'img_cx', 'img_cy']


# ---------------------------------------------------------------- solar-physics descriptors (ablation only)

def _axis(xs, ys):
    """orientation (deg, image y up, -90..90, positive = right end up) and elongation of a pixel set."""
    xm, ym = xs.mean(), ys.mean()
    sxx, syy, sxy = ((xs - xm) ** 2).mean(), ((ys - ym) ** 2).mean(), ((xs - xm) * (ys - ym)).mean()
    th = 0.5 * np.degrees(np.arctan2(-2 * sxy, sxx - syy))
    tr_, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(max(tr_ * tr_ / 4 - det, 0.))
    return float(th), float(np.sqrt(max(tr_ / 2 + disc, 1e-3) / max(tr_ / 2 - disc, 1e-3))), xm, ym


def _adiff(a, b):
    return float(abs((a - b + 90) % 180 - 90))


def physics(sub, xs, ys, cx, cy, major, a, dcx, dcy, dR):
    """Solar-physics descriptors of one candidate: position on the disk (latitude, central-meridian distance,
    mu = cos(heliocentric angle), B0 = 0 assumed), tilt mirrored by hemisphere, foreshortening-corrected size, and the
    linkage geometry of multi-fragment candidates (gaps, collinearity, tilt difference)."""
    ux, uy = (cx - dcx) / dR, (dcy - cy) / dR
    r = float(np.hypot(ux, uy)); mu_ = float(np.sqrt(max(1. - r * r, 0.02)))
    lat = float(np.degrees(np.arcsin(np.clip(uy, -1, 1))))
    lon = float(np.degrees(np.arcsin(np.clip(ux / max(np.cos(np.radians(lat)), 0.05), -1, 1))))
    th, el, xm, ym = _axis(xs, ys)
    rad = _adiff(th, np.degrees(np.arctan2(uy, ux))) if r > 0.02 else 45.
    cr = np.cos(np.radians(rad)); sr = np.sin(np.radians(rad))
    f = dict(ph_lat=lat, ph_abslat=abs(lat), ph_lon=lon, ph_abslon=abs(lon), ph_mu=mu_, ph_r=r, ph_tilt=th, ph_tilt_h=th * (1. if lat >= 0 else -1.),
             ph_abstilt=abs(th), ph_rad_ang=rad, ph_llen=float(np.log(max(major, 1.) * np.sqrt(cr * cr / mu_ ** 2 + sr * sr))), ph_larea=float(np.log(a / mu_)))
    lk = dict(lk_n=1, lk_gap_max=0., lk_gap_mean=0., lk_gap_rel=0., lk_gap_dep=0., lk_second=0., lk_align=0., lk_tiltdiff=0., lk_elong1=el, lk_fit=0., lk_fit_rel=0.)
    t = np.radians(th); px = (xs - xm) * np.cos(t) - (ys - ym) * np.sin(t); py = (xs - xm) * np.sin(t) + (ys - ym) * np.cos(t)
    step = max(len(px) // 1500, 1)
    try:
        co = np.polyfit(px[::step], py[::step], 2)
        res = float(np.sqrt(np.mean((py[::step] - np.polyval(co, px[::step])) ** 2)))
    except Exception:
        res = 0.
    lk['lk_fit'] = res; lk['lk_fit_rel'] = float(res / max(a / max(major, 1.), 1.))
    n, lab, st, cen = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 20]
    if len(ids) >= 2:
        ids = sorted(ids, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:8]
        D = np.zeros((len(ids), len(ids)))
        for p_, i in enumerate(ids):
            dt = cv2.distanceTransform((lab != i).astype(np.uint8), cv2.DIST_L2, 3)
            for q_, j in enumerate(ids):
                if p_ != q_:
                    D[p_, q_] = dt[lab == j].min()
        D = np.minimum(D, D.T)
        inn = [0]; edges = []
        while len(inn) < len(ids):
            out_ = [k for k in range(len(ids)) if k not in inn]
            sd_ = D[np.ix_(inn, out_)]
            p_, q_ = np.unravel_index(sd_.argmin(), sd_.shape)
            edges.append(sd_[p_, q_]); inn.append(out_[q_])
        y1, x1 = np.nonzero(lab == ids[0]); y2, x2 = np.nonzero(lab == ids[1])
        t1, e1, xm1, ym1 = _axis(x1, y1); t2, e2, xm2, ym2 = _axis(x2, y2)
        link = np.degrees(np.arctan2(-(ym2 - ym1), xm2 - xm1))
        gmax = float(max(edges))
        lk.update(lk_n=len(ids), lk_gap_max=gmax, lk_gap_mean=float(np.mean(edges)), lk_gap_rel=float(gmax / max(major, 1.)),
                  lk_gap_dep=float(gmax / mu_), lk_second=float(st[ids[1], cv2.CC_STAT_AREA] / a), lk_align=_adiff(t1, link),
                  lk_tiltdiff=_adiff(t1, t2) if e2 > 1.5 else 45., lk_elong1=e1)
    f.update(lk)
    return f


# ---------------------------------------------------------------- candidate pool

def candidates(prob, std, spine, im):
    """All candidate instances of one image: list of (rle, feature dict).

    prob: calibrated ensemble map (uint8, 2048 or 1024 px); std: member disagreement (uint8, 1024 px);
    spine: calibrated spine map of member D (uint8, 1024 px) or None; im: the image (uint8, 2048 px).
    """
    p = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    sd = cv2.resize(std, (W, H), interpolation=cv2.INTER_LINEAR)
    sp = cv2.resize(spine, (W, H), interpolation=cv2.INTER_LINEAR) if spine is not None else None
    im = im.astype(np.float32)
    disk = im[im > 20]; med = float(np.median(disk)) if disk.size else 128.
    dm = cv2.medianBlur(im.astype(np.uint8), 5) > 25                 # solar disk: centre and radius from the image itself
    ysd = np.where(dm.sum(1) > 20)[0]; xsd = np.where(dm.sum(0) > 20)[0]
    if len(ysd) > 100 and len(xsd) > 100:
        dcx, dcy, dR = (xsd[0] + xsd[-1]) / 2, (ysd[0] + ysd[-1]) / 2, ((xsd[-1] - xsd[0]) + (ysd[-1] - ysd[0])) / 4
    else:
        dcx, dcy, dR = W / 2, H / 2, 0.45 * W
    out = []
    full = np.zeros((H, W), np.uint8, order='F')
    lo = (p >= 0.1 * 255).astype(np.uint8)
    raw, masks = {}, {}

    def describe(sub, x0, y0, w, h, m0, t, c, g):
        a = int(sub.sum())
        if a < 150:
            return
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
        ph = physics(sub, xs, ys, cx, cy, major, a, dcx, dcy, dR)
        ph.update(dark_frac=float((iv < 0.55).mean()), ring_bright=float((rv > 1.12).mean()) if rv.size else 0.,
                  ring_p5=float(np.percentile(rv, 5)) if rv.size else 1.)
        nf, _, stf, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
        fr = stf[1:, cv2.CC_STAT_AREA]; fr = fr[fr >= 20]
        f = dict(
            src=0, src_score=0., t=t, close=c, group=g, fine=int(t in TS_FINE), larea=np.log(fa), w=w, h=h, pmax=pv.max(), pmean=pv.mean(), pmed=np.median(pv),
            p10=np.percentile(pv, 10), p90=np.percentile(pv, 90), frac5=(pv > .5).mean(), frac7=(pv > .7).mean(),
            pring=pring.mean() if pring.size else 0., smean=dv.mean(), s90=np.percentile(dv, 90),
            imean=iv.mean(), ring=rv.mean() if rv.size else 1., contrast=(rv.mean() - iv.mean()) if rv.size else 0.,
            imin=np.percentile(iv, 5), solidity=a / ha, compact=4 * np.pi * a / max(peri, 1.) ** 2, elong=elong,
            width=a / max(major, 1.), rdisk=np.hypot(cx - W / 2, cy - H / 2) / (W / 2), hole_frac=(fa - a) / fa,
            frag_n=int(len(fr)), frag_max=float(fr.max() / a) if len(fr) else 0., core=float(core.sum() / a))
        f.update(ph)
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

    for t, c, g in SETTINGS:
        if t not in raw:
            raw[t] = (p >= t * 255).astype(np.uint8)
        m0 = raw[t]
        if (t, c) not in masks:
            masks[(t, c)] = (cv2.morphologyEx(m0, cv2.MORPH_CLOSE, KER[c]) & lo | m0) if c else m0
        m = masks[(t, c)]
        grouped = g != c
        if grouped:
            sm = m.reshape(H // 4, 4, W // 4, 4).max((1, 3))
            n, gl, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(sm, cv2.MORPH_CLOSE, KERG[g]), connectivity=8)
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
            describe(sub, x0, y0, w, h, m0, t, c, g)
    n, lab, st, cen = cv2.connectedComponentsWithStats((p >= 0.5 * 255).astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    bigc = areas[areas >= 600]
    for _, f in out:
        f['img_n'] = int(len(bigc)); f['img_larea'] = float(np.log1p(bigc.sum()))
        f['rel_area'] = float(np.exp(f['larea']) / max(bigc.max(), 1)) if len(bigc) else 0.
        f['img_spread'] = float(sd.mean())
    return out


def overlaps(cs):
    """Nesting descriptors (in place) and the boolean overlap matrix of the candidates of one image."""
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


# ---------------------------------------------------------------- simulated annotators

def sim_features(cs, lev, team):
    """Descriptors of every candidate against the simulated annotators.

    lev: [A, 1024, 1024] uint8 levels 0..3 of every annotator's map (probability >= 0.3 / 0.5 / 0.7);
    team: 0-based indices of the annotators who worked in teams (the test set is team-annotated).
    """
    NA = lev.shape[0]
    LAB = {}
    for L in (1, 2, 3):
        for a in range(NA):
            n, lab, st, _ = cv2.connectedComponentsWithStats((lev[a] >= L).astype(np.uint8), connectivity=8)
            LAB[(L, a)] = (lab, st[:, cv2.CC_STAT_AREA].astype(np.float64))
    nper = np.array([(LAB[(2, a)][1][1:] >= 40).sum() for a in range(NA)], float)      # >= 160 px at 2048
    rows = []
    for r, f in cs:
        x0, y0, w, h = [int(v) for v in mu.toBbox(r)]
        m = mu.decode(r)[y0:y0 + h, x0:x0 + w]
        ya, xa = y0 // 2, x0 // 2
        hh, ww = (y0 + h + 1) // 2 - ya, (x0 + w + 1) // 2 - xa
        mm = np.zeros((hh * 2, ww * 2), np.uint8); mm[y0 - ya * 2:y0 - ya * 2 + h, x0 - xa * 2:x0 - xa * 2 + w] = m
        ms = mm.reshape(hh, 2, ww, 2).max((1, 3)).astype(bool)
        ac = float(ms.sum())
        d = {}
        for L, tag in ((2, ''), (1, '_lo'), (3, '_hi')):
            iou = np.zeros(NA); cov = np.zeros(NA); nt = np.zeros(NA); over = np.full(NA, np.nan)
            for a in range(NA):
                lab, area = LAB[(L, a)]
                ids = lab[ya:ya + hh, xa:xa + ww][ms]
                if not ids.size or not ids.any():
                    continue
                cnt = np.bincount(ids, minlength=len(area))[1:].astype(np.float64)
                b = int(cnt.argmax()); inter = cnt[b]
                iou[a] = inter / (ac + area[b + 1] - inter); cov[a] = cnt.sum() / ac
                nt[a] = (cnt >= max(5., 0.05 * ac)).sum(); over[a] = (area[b + 1] - inter) / area[b + 1]
            hit = iou > .5
            d['sim_q' + tag] = float((iou * hit).mean()); d['sim_m' + tag] = float(hit.mean())
            if L == 2:
                d.update(sim_iou=float(iou.mean()), sim_iou_sd=float(iou.std()), sim_cov=float(cov.mean()), sim_any=float((cov > .3).mean()),
                         sim_nt=float(nt.mean()), sim_over=float(np.nanmean(over)) if np.isfinite(over).any() else 1.,
                         sim_q_team=float((iou * hit)[team].mean()), sim_m_team=float(hit[team].mean()),
                         sim_q_solo=float(np.delete(iou * hit, team).mean()) if len(team) < NA else 0.,
                         sim_iou_max=float(iou.max()), sim_iou_q25=float(np.percentile(iou, 25)))
        d['sim_qmax'] = max(d['sim_q'], d['sim_q_lo'], d['sim_q_hi'])
        d['sim_n_img'] = float(nper.mean()); d['sim_n_sd'] = float(nper.std())
        rows.append(d)
    for (r, f), d in zip(cs, rows):
        f.update(d)


# ---------------------------------------------------------------- detector evidence

DETF = ['det_iou', 'det_score', 'det_q', 'det_iou5', 'det_cov', 'det_n', 'det_smax', 'det_in', 'det_nimg']


def det_desc(rles, dets):
    """Descriptors of the candidates of one image against one list of detector instances [(counts, score)]."""
    n = len(rles)
    out = {k: np.zeros(n) for k in DETF}
    if not n or not dets:
        return out
    dr = [dict(size=[H, W], counts=c.encode()) for c, _ in dets]; sc = np.array([s for _, s in dets])
    iou = np.array(mu.iou(rles, dr, [0] * len(dr))).reshape(n, len(dr))
    cov = np.array(mu.iou(rles, dr, [1] * len(dr))).reshape(n, len(dr))        # intersection / candidate area
    j = iou.argmax(1); best = iou.max(1)
    out['det_iou'] = best; out['det_score'] = sc[j] * (best > 0)
    out['det_q'] = (iou * (iou > .5) * sc[None]).max(1)
    hi = sc >= .5
    if hi.any():
        out['det_iou5'] = iou[:, hi].max(1)
    mid = sc >= .3
    if mid.any():
        U = mu.merge([d for d, m in zip(dr, mid) if m])
        out['det_cov'] = np.array(mu.iou(rles, [U], [1])).reshape(n)
    out['det_n'] = ((cov >= .1) & mid[None]).sum(1).astype(float)
    out['det_smax'] = np.where(cov >= .5, sc[None], 0.).max(1)
    cj = cov[np.arange(n), j]                                                        # share of the best instance inside the candidate
    with np.errstate(divide='ignore', invalid='ignore'):
        out['det_in'] = np.where(best > 0, 1. / (1. / np.maximum(best, 1e-9) - 1. / np.maximum(cj, 1e-9) + 1.), 0.)
    out['det_nimg'] = np.full(n, float(hi.sum()))
    return out


def det_features(cs, dets):
    """dets: {tag: {fold: [views: [(counts, score)]]}} for this image (tags m31, m34; four views each).
    Adds the groups d2 (m31), d3 (m34) and d23 (both): every descriptor averaged over the fold models and then over
    the views and detectors of the group, plus the spread of the best IoU and the minimum matched score."""
    rles = [c[0] for c in cs]
    per = {}
    for tag, by_fold in dets.items():
        nv = len(next(iter(by_fold.values())))
        for v in range(nv):
            ds = [det_desc(rles, by_fold[f][v]) for f in sorted(by_fold)]
            per[(tag, v)] = {k: np.mean([x[k] for x in ds], 0) for k in DETF}
    groups = {'d2': [m for m in per if m[0] == 'm31'], 'd3': [m for m in per if m[0] == 'm34']}
    groups['d23'] = groups['d2'] + groups['d3']
    d = {}
    for g, members in groups.items():
        if not members:
            continue
        for k in DETF:
            d[k.replace('det_', g + '_')] = np.mean([per[m][k] for m in members], 0)
        if len(members) > 1:
            d[g + '_iou_sd'] = np.std([per[m]['det_iou'] for m in members], 0)
            d[g + '_q_min'] = np.min([per[m]['det_q'] for m in members], 0)
    for i, (r, f) in enumerate(cs):
        for key, arr in d.items():
            f[key] = float(arr[i])


def greedy(pred, ov, lam):
    """Pixel-disjoint selection: candidates in decreasing predicted quality, skipping those that overlap a taken one,
    until the predicted quality drops to lam."""
    order = np.argsort(-pred); taken = np.zeros(len(pred), bool); blocked = np.zeros(len(pred), bool)
    for i in order:
        if pred[i] <= lam:
            break
        if blocked[i]:
            continue
        taken[i] = True; blocked |= ov[i]
    return taken
