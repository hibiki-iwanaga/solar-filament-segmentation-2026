"""Network definitions, weight loading and test-time prediction of every network of the pipeline.

Dense probability maps (soft annotator-mean mask):
  B  U-Net, EfficientNet-B4 encoder, image at 1024 px                       (training/s02_dense_b.py)
  C  U-Net, ResNet34, image + limb-flattened image, trained on 1024 px crops of the native 2048 px frames,
     applied at 2048 px                                                    (training/s03_dense_c.py)
  D  UNet++, ResNet34, image + flattened image at 1024 px, mask and spine outputs (training/s04_dense_d.py)
  E  annotator-conditioned U-Net (ResNet34 + FiLM by an annotator embedding): one map per annotator, used to
     describe how many simulated annotators would draw a candidate          (training/s05_sim_annotators.py)
Instance detectors (evidence for the candidate descriptors):
  m31  Mask R-CNN v2 (ResNet50-FPN), image at 1024 px                       (training/s12_detector_1024.py)
  m34  Mask R-CNN v2, trained on 1024 px crops, applied to the 2048 px frame (training/s13_detector_2048.py)
Stage-2 refiners (redraw one selected instance in an oriented 384 x 128 crop, 6 input channels):
  v17 UNet++ ResNet34, v18 U-Net ResNeXt50-32x4d, v19 U-Net EfficientNet-B4  (training/s09-s11_refiner_*.py)

All networks are trained with 5 folds; the test predictions average the five fold models.
Weights are stored in float16 and evaluated in float32 (with autocast on a GPU).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pycocotools import mask as mu

from .preprocess import norm

S1 = 1024
H = W = 2048


def load_state(path):
    """State dict with floating-point tensors rounded through float16 (the precision the weights are stored in)."""
    sd = torch.load(path, map_location='cpu')
    return {k: (v.half().float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


def save_half(state, path):
    torch.save({k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in state.items()}, path)


def _ac(dev):
    return torch.autocast(dev.type, enabled=dev.type == 'cuda')


# ---------------------------------------------------------------- dense members B, C, D

def build_dense(member):
    if member == 'B':
        return smp.Unet('efficientnet-b4', encoder_weights=None, in_channels=1, classes=1)
    if member == 'C':
        return smp.Unet('resnet34', encoder_weights=None, in_channels=2, classes=1)
    if member == 'D':
        return smp.UnetPlusPlus('resnet34', encoder_weights=None, in_channels=2, classes=2)
    raise ValueError(member)


@torch.no_grad()
def predict_b(model, im1k, med, dev):
    """Member B: 1024 px image, flips (4 views) -> uint8 map at 1024 px."""
    x = torch.from_numpy(norm(im1k, med))[None, None].to(dev)
    acc = 0
    for d in [None, (3,), (2,), (2, 3)]:
        with _ac(dev):
            p = torch.sigmoid(model(x.flip(d) if d else x)).float()
        acc = acc + (p.flip(d) if d else p)
    acc = acc / 4
    return (acc[0, 0] * 255).byte().cpu().numpy()


@torch.no_grad()
def predict_c(model, im2k, flat2k, med, dev):
    """Member C: 2048 px image + flattened image, 8 dihedral views -> (uint8 map at 2048 px, area-averaged 1024 px map)."""
    x = torch.from_numpy(np.stack([norm(im2k, med), flat2k.astype(np.float32) / 255.]))[None].to(dev)
    acc = 0
    for k in range(4):
        for fl in (False, True):
            xx = torch.rot90(x, k, (2, 3))
            if fl:
                xx = xx.flip(3)
            with _ac(dev):
                p = torch.sigmoid(model(xx)).float()
            if fl:
                p = p.flip(3)
            acc = acc + torch.rot90(p, -k, (2, 3))
    acc = acc / 8
    o2 = (acc[0, 0] * 255).byte().cpu().numpy()
    o1 = (F.interpolate(acc, size=(S1, S1), mode='area')[0, 0] * 255).byte().cpu().numpy()
    return o2, o1


@torch.no_grad()
def predict_d(model, x2, med, dev):
    """Member D: [image, flattened image] at 1024 px, 8 dihedral views in one batch -> uint8 [mask, spine] at 1024 px."""
    x = np.stack([norm(x2[0], med), x2[1].astype(np.float32) / 255.])
    x = torch.from_numpy(x)[None].to(dev)
    views = []
    for k in range(4):
        for fl in (False, True):
            xx = torch.rot90(x, k, (2, 3))
            views.append(xx.flip(3) if fl else xx)
    with _ac(dev):
        if dev.type == 'cuda':
            p = torch.sigmoid(model(torch.cat(views))).float()
        else:                                     # one view at a time on a CPU (memory)
            p = torch.cat([torch.sigmoid(model(xx)).float() for xx in views])
    acc = 0
    v = 0
    for k in range(4):
        for fl in (False, True):
            q = p[v:v + 1]
            if fl:
                q = q.flip(3)
            acc = acc + torch.rot90(q, -k, (2, 3))
            v += 1
    return (acc[0] / 8 * 255).byte().cpu().numpy()


# ---------------------------------------------------------------- simulated annotators E

class CondUnet(nn.Module):
    """U-Net whose encoder features and decoder output are modulated by an annotator embedding (index 0 = unknown)."""
    def __init__(self, n_ann, emb=16):
        super().__init__()
        self.net = smp.Unet('resnet34', encoder_weights=None, in_channels=2, classes=2)
        chs = list(self.net.encoder.out_channels)[1:]
        dch = self.net.segmentation_head[0].in_channels
        self.emb = nn.Embedding(n_ann + 1, emb)
        self.film = nn.ModuleList([nn.Linear(emb, 2 * c) for c in chs] + [nn.Linear(emb, 2 * dch)])
        self.net.eval()
        try:                               # smp <= 0.3 takes the features unpacked, newer versions as one list
            with torch.no_grad():
                self.net.decoder(*[torch.zeros(1, c, 64 >> k, 64 >> k) for k, c in enumerate(self.net.encoder.out_channels)])
            self.star = True
        except TypeError:
            self.star = False

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


@torch.no_grad()
def predict_annotators(model, x2, med, n_ann, dev):
    """Mask probability of every annotator (1..n_ann) for one image, identity and horizontal-flip views: [A, S, S]."""
    x = torch.from_numpy(np.stack([norm(x2[0], med), x2[1].astype(np.float32) / 255.]))[None].to(dev)
    x = torch.cat([x, x.flip(3)])
    with _ac(dev):
        feats = model.encode(x)
    S = x.shape[-1]
    pa = torch.zeros(n_ann, S, S, device=dev)
    ids = torch.arange(1, n_ann + 1, device=dev)
    for v in (0, 1):
        fv = [f[v:v + 1] for f in feats]
        for ch in ids.split(8):
            with _ac(dev):
                q = torch.sigmoid(model.decode([f.expand(len(ch), -1, -1, -1) for f in fv], ch)[:, 0]).float()
            pa[ch - 1] += q.flip(2) if v == 1 else q
    return pa / 2


# ---------------------------------------------------------------- Mask R-CNN detectors

VIEWS = ((0, False), (0, True), (2, True), (1, False))      # (quarter turns, horizontal flip): identity, h-flip, v-flip, 90 degrees


def build_detector(train_size):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    kw = dict(min_size=train_size, max_size=train_size, box_detections_per_img=100, box_score_thresh=0.05,
              image_mean=[0.45, 0.45, 0.45], image_std=[0.22, 0.22, 0.22])
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kw)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return model


def _fwd(a, view):
    k, fl = view
    a = np.rot90(a, k, (-2, -1)) if k else a
    return a[..., ::-1] if fl else a


def _inv(a, view):
    k, fl = view
    a = a[..., ::-1] if fl else a
    return np.rot90(a, -k, (-2, -1)) if k else a


def _to_input(x2):
    """[2, S, S] uint8 (image, flattened image) -> [3, S, S] float in [0, 1]."""
    a = torch.from_numpy(np.ascontiguousarray(x2)).float() / 255.
    return torch.stack([a[0], a[1], a[1]])


@torch.no_grad()
def detect(model, x2, view, native, dev):
    """Instances of one image seen in one view, mapped back to the 2048 px frame: [(RLE counts, score)].

    native=False (m31): x2 at 1024 px, soft masks resized to 2048 px and thresholded at 0.5.
    native=True (m34): x2 at 2048 px, the full frame is fed at its native size, masks thresholded at 0.5.
    """
    model.eval()
    if native:
        model.transform.min_size = (x2.shape[-1],)
        model.transform.max_size = x2.shape[-1]
    with _ac(dev):
        o = model([_to_input(np.ascontiguousarray(_fwd(x2, view))).to(dev)])[0]
    sc = o['scores'].float().cpu().numpy()
    if native:
        mk = (o['masks'][:, 0] >= 0.5).to(torch.uint8).cpu().numpy()
    else:
        mk = o['masks'][:, 0].float().cpu().numpy()
    del o
    out = []
    import cv2
    for s, m in zip(sc, mk):
        m = np.ascontiguousarray(_inv(m, view))
        big = m if native else (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5)
        if big.sum() < 40:
            continue
        out.append((mu.encode(np.asfortranarray(big.astype(np.uint8)))['counts'].decode(), float(s)))
    return out


# ---------------------------------------------------------------- stage-2 refiners

REFINERS = {'v17': ('UnetPlusPlus', 'resnet34'), 'v18': ('Unet', 'resnext50_32x4d'), 'v19': ('Unet', 'efficientnet-b4')}


def build_refiner(tag):
    arch, enc = REFINERS[tag]
    return getattr(smp, arch)(enc, encoder_weights=None, in_channels=6, classes=1)


@torch.no_grad()
def predict_refiner(model, crops, dev, bs=64):
    """crops [N, 6, 128, 384] uint8 -> refined probability [N, 128, 384] uint8 (4 flip views)."""
    model.eval()
    bs = bs if dev.type == 'cuda' else 8            # memory on a CPU
    res = np.zeros((len(crops), crops.shape[2], crops.shape[3]), np.uint8)
    for b in range(0, len(crops), bs):
        x = torch.from_numpy(np.asarray(crops[b:b + bs]).astype(np.float32) / 255.).to(dev)
        acc = 0
        for fh in (False, True):
            for fv in (False, True):
                xx = x.flip(3) if fh else x
                xx = xx.flip(2) if fv else xx
                with _ac(dev):
                    p = torch.sigmoid(model(xx)).float()
                p = p.flip(2) if fv else p
                acc = acc + (p.flip(3) if fh else p)
        res[b:b + bs] = (acc[:, 0] / 4 * 255).byte().cpu().numpy()
    return res
