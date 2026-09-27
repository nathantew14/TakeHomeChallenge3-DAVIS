"""Evaluate binary foreground masks. Requires Python 3, Pillow and NumPy."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def read_mask(path):
    with Image.open(path) as im:
        if im.mode not in ('1', 'L', 'P'):
            raise ValueError(f'{path}: use a single-channel binary PNG, not an RGB overlay.')
        a = np.asarray(im)
    values = set(np.unique(a).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f'{path}: mask must contain only 0/1 or 0/255; got {sorted(values)[:12]}')
    return a > 0

def metrics(pred, truth):
    if pred.shape != truth.shape:
        raise ValueError(f'Mask dimensions differ: {pred.shape} vs {truth.shape}. Export at the supplied image size.')
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    union = tp + fp + fn
    return {'iou': tp / union if union else 1.0, 'tp': tp, 'fp': fp, 'fn': fn}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', required=True, type=Path)
    parser.add_argument('--pred', required=True, nargs='+', type=Path)
    args = parser.parse_args()
    gt = read_mask(args.gt)
    print('prediction,IoU,TP,FP,FN')
    for path in args.pred:
        r = metrics(read_mask(path), gt)
        print(f'{path.name},{r["iou"]:.6f},{r["tp"]},{r["fp"]},{r["fn"]}')
if __name__ == '__main__':
    main()
