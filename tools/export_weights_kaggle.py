"""How `weights/` was built on Kaggle (first cells of the inference notebook): the fold models of every network are
read from the outputs of the training notebooks, stored in float16 in the repository layout, and the LightGBM models
and calibration histograms of stages 14 and 16 are copied. Run on Kaggle with those notebooks attached as inputs."""
import glob
import json
import os
import pickle
import shutil

import torch

from solfil.nets import save_half
from solfil.pipeline import WEIGHTS

SRC = {'B': ('filament-v5b-effb4-5fold', 'unet_f{f}.pt'), 'C': ('filament-v12-flat-resnet34', 'unet_f{f}.pt'),
       'D': ('filament-v14-unetpp-spine', 'unetpp_f{f}.pt'), 'E': ('filament-v21-annotator-unet', 'cond_f{f}.pt'),
       'm31': ('filament-v31-maskrcnn2-views', 'det2_f{f}.pt'), 'm34': ('filament-v34-maskrcnn2-native', 'det3_f{f}.pt'),
       'v17': ('filament-v17-refiner-2src', 'refiner_f{f}.pt'), 'v18': ('filament-v18-refiner-ens', 'refiner_f{f}.pt'),
       'v19': ('filament-v19-lowlam-refiners', 'refiner_f{f}.pt')}


def find(slug, name):
    c = [p for p in glob.glob(f'/kaggle/input/**/{name}', recursive=True) if f'/{slug}/' in p and '/prev_' not in p]
    assert len(c) == 1, (slug, name, c)
    return c[0]


def main(wd='weights'):
    for key, (slug, pat) in SRC.items():
        os.makedirs(f'{wd}/{WEIGHTS[key]}', exist_ok=True)
        for f in range(5):
            save_half(torch.load(find(slug, pat.format(f=f)), map_location='cpu'), f'{wd}/{WEIGHTS[key]}/fold{f}.pt')
    a21 = pickle.load(open(find('filament-v21-annotator-unet', 'annotators21.pkl'), 'rb'))
    json.dump(dict(anns=[str(a) for a in a21['anns']], team=[int(t) for t in a21['team']]), open(f'{wd}/{WEIGHTS["E"]}/annotators.json', 'w'))
    os.makedirs(f'{wd}/stage1', exist_ok=True)
    os.makedirs(f'{wd}/stage2', exist_ok=True)
    for nm in ('reg.txt', 'hit.txt', 'iou.txt', 'model.json'):
        shutil.copy(find('filament-s14-pool-quality-final', f'stage1_models/{nm}'), f'{wd}/stage1/{nm}')
    shutil.copy(find('filament-s14-pool-quality-final', 'calibration.json'), f'{wd}/stage1/calibration.json')
    for nm in ('seed0.txt', 'seed1.txt', 'seed2.txt', 'model.json'):
        shutil.copy(find('filament-s16-rescore-final', f'stage2_models/{nm}'), f'{wd}/stage2/{nm}')


if __name__ == '__main__':
    main()
