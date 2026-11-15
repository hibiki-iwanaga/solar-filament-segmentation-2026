"""Draws report/figures/pipeline.png (the data pipeline) with matplotlib."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'pipeline.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
C_IN, C_NET, C_DEC, C_OUT = '#e8eef7', '#fde9d9', '#e4f2e1', '#f2e1ee'


def box(x, y, w, h, text, color, fs=5.7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=color, ec='#555555', lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', linespacing=1.15)


def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#444444', shrinkA=0, shrinkB=0))


# input and preprocessing
box(1, 44, 15, 12, 'GONG H-alpha\n2048 x 2048\n(test image)', C_IN, bold=True)
box(1, 26, 15, 13, 'image / disk median\nlimb-flattened image\n(radial median profile)\n1024 px copies', C_IN)
arrow(8.5, 44, 8.5, 39.3)
# networks
box(21, 47, 22, 10, 'dense maps (5 folds each)\nB U-Net EffNet-B4 1024 px\nC U-Net R34 2048 px\nD UNet++ R34 + spine', C_NET)
box(21, 33.5, 22, 10, 'simulated annotators E\nU-Net R34 + FiLM embedding\none map per annotator (31)', C_NET)
box(21, 20, 22, 10, 'Mask R-CNN v2 (5 folds)\nm31 at 1024 px, m34 at 2048 px\n4 test-time views each', C_NET)
for y in (52, 38.5, 25):
    arrow(16.3, 32.5, 20.7, y)
# stage 14
box(48, 44.5, 22, 12.5, 'calibrated ensemble B+C+D\n(quantile match to OOF)\ncandidate pool: 5 thresholds\nx closing / grouping\n(~157 candidates / image)', C_DEC)
box(48, 20, 22, 21.5, 'candidate descriptors\nmap, shape, contrast, spine,\nnesting, simulated\nannotators (match rate),\ndetector agreement\n\nLightGBM quality model\n(regression + hurdle)\ngreedy pixel-disjoint selection', C_DEC)
arrow(43.3, 52, 47.7, 51)
arrow(43.3, 38.5, 47.7, 33)
arrow(43.3, 25, 47.7, 27)
arrow(59, 44.5, 59, 41.8)
# stage 15/16
box(75, 40, 24, 17, 'stage-2 refiners (5 folds each)\noriented 384 x 128 crop,\n6 channels (image, flat, map,\nspine, seed, other seeds)\nUNet++ R34 | U-Net RX50 |\nU-Net EffNet-B4 (mean)', C_NET)
box(75, 20, 24, 16.5, 'second LightGBM on the\nrefined outlines\n(matched IoU of the outline);\npaste back in decreasing\nquality (pixel-disjoint),\nkeep q > 0.20, p >= 0.5', C_DEC)
arrow(70.3, 33, 74.7, 45)
arrow(87, 40, 87, 36.8)
box(75, 3, 24, 11, 'submission\n(instance RLEs,\n6.9 filaments / image)', C_OUT, bold=True)
arrow(87, 20, 87, 14.3)
ax.text(32, 17.5, 'networks', ha='center', fontsize=6.4, style='italic', color='#555555')
ax.text(59, 17.5, 'stage 1: candidates and selection', ha='center', fontsize=6.4, style='italic', color='#555555')
ax.text(87, 58.6, 'stage 2: refinement', ha='center', fontsize=6.4, style='italic', color='#555555')
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print('written', OUT)
