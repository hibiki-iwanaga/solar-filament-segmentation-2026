"""Run one training stage outside Kaggle.

The files in `training/` are the sources of the Kaggle notebooks that produced the weights, in "percent" format
(`# %%` starts a cell; a cell that starts with `%%writefile PATH` is written to PATH instead of being executed).
They read their inputs from `/kaggle/input/**` and write to `/kaggle/working`. This runner executes a stage cell by
cell after replacing those two roots:

    python tools/run_stage.py training/s14_pool_quality.py --input /data/fil/input --work /data/fil/work/s14

`--input` must contain the competition data (folder MAGFiLO_1.0_Kaggle_2026) and the output folders of the earlier
stages under their Kaggle notebook names (e.g. filament-v5b-effb4-5fold/), because the stages find their inputs by
those names. Stages 14-16 read the code of earlier notebooks from their `__notebook__.ipynb`; the files here are
already the fully patched sources, so that step is not needed.
"""
import argparse
import os
import sys


def cells(src):
    """Code cells of a percent-format script (markdown cells are skipped)."""
    out = []
    for chunk in ('\n' + src).split('\n# %%'):
        chunk = chunk.strip('\n')
        if chunk and not chunk.startswith(' [markdown]'):
            out.append(chunk)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', help='path of a training/*.py stage file')
    ap.add_argument('--input', default='/kaggle/input', help='replaces /kaggle/input')
    ap.add_argument('--work', default='/kaggle/working', help='replaces /kaggle/working')
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    src = open(a.stage).read().replace('/kaggle/input', a.input.rstrip('/')).replace('/kaggle/working', a.work.rstrip('/'))
    g = globals()          # cells run in __main__, so functions defined in them can be sent to worker processes
    for c in cells(src):
        if c.startswith('%%writefile'):
            head, body = c.split('\n', 1)
            open(head.split()[1], 'w').write(body + '\n')
            print('written', head.split()[1], flush=True)
        else:
            exec(compile(c, a.stage, 'exec'), g)


if __name__ == '__main__':
    sys.exit(main())
