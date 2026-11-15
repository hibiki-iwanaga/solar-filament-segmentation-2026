"""Draws report/figures/eval.png from the out-of-fold evaluation written by stage 16 (eval_oof.json)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'eval_oof.json')))
OUT = os.path.join(HERE, 'figures', 'eval.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
# (a) best IoU of every annotated filament
h = np.array(EV['best_iou_hist'], float); e = np.linspace(0, 1, len(h) + 1)
ax[0].bar(e[:-1], h / h.sum(), width=e[1] - e[0], align='edge', color=['#c9c9c9' if x < 0.5 else '#3b75af' for x in e[:-1]], ec='white', lw=0.4)
ax[0].axvline(0.5, color='#aa3333', lw=0.8, ls='--')
ax[0].set_xlabel('best IoU of an annotated filament'); ax[0].set_ylabel('share')
ax[0].set_title(f"(a) matched (IoU > 0.5): {EV['all']['TP'] / (EV['all']['TP'] + EV['all']['FN']):.1%}")
# (b) pixel Dice per annotator record
h = np.array(EV['record_dice_hist'], float); e = np.linspace(0, 1, len(h) + 1)
ax[1].bar(e[:-1], h / h.sum(), width=e[1] - e[0], align='edge', color='#5a9e5a', ec='white', lw=0.4)
ax[1].axvline(EV['pixel_dice_per_record']['mean'], color='k', lw=0.8, ls='--')
ax[1].set_xlabel('pixel Dice of an annotator record'); ax[1].set_title(f"(b) mean {EV['pixel_dice_per_record']['mean']:.3f}")
# (c) PQ by fold and by number of records, with the bootstrap interval
names = [f'fold {f}' for f in sorted(EV['PQ_by_fold'], key=int)] + ['K = 1', 'K >= 2', 'all']
vals = [EV['PQ_by_fold'][f] for f in sorted(EV['PQ_by_fold'], key=int)] + [EV['K1']['PQ'], EV['K2+']['PQ'], EV['all']['PQ']]
cols = ['#8fb3d9'] * 5 + ['#e0a458', '#e0a458', '#3b75af']
ax[2].bar(range(len(vals)), vals, color=cols)
lo, hi = EV['PQ_bootstrap_95']
ax[2].errorbar([len(vals) - 1], [EV['all']['PQ']], yerr=[[EV['all']['PQ'] - lo], [hi - EV['all']['PQ']]], color='k', capsize=2, lw=0.8)
ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(names, rotation=45, ha='right')
ax[2].set_ylim(0.35, max(vals) + 0.04); ax[2].set_ylabel('PQ (out of fold)'); ax[2].set_title('(c) PQ by fold / records per image')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT, dpi=300); plt.close()
print('written', OUT)
