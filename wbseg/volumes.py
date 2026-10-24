"""Structure volumes from label maps, and agreement between two sets of volumes."""

import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd

from .data import subject_id
from .labels import NAMES


def voxel_volume_ml(img):
    """Volume of one voxel in millilitres, from the image header."""
    return float(np.prod(img.header.get_zooms()[:3])) / 1000.0


def label_volumes(path, labels):
    """{label: volume in ml} for one segmentation file."""
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.int64)
    counts = np.bincount(data[data >= 0].ravel())
    vox = voxel_volume_ml(img)
    return {int(l): (counts[l] if l < len(counts) else 0) * vox for l in labels if l != 0}


def volume_table(folder, labels, pattern="*.nii*"):
    """One row per file, one column per structure (ml). Works on predictions or FreeSurfer asegs."""
    rows = []
    for path in sorted(glob(os.path.join(folder, pattern))):
        vols = label_volumes(path, labels)
        rows.append({"id": subject_id(path), **{NAMES[l]: v for l, v in vols.items()}})
    return pd.DataFrame(rows)


def agreement(results):
    """Per-structure agreement between predicted and FreeSurfer volumes.

    `results` is the long table from evaluate(). Returns Pearson r, the mean
    difference (predicted - FreeSurfer) in percent, and the mean absolute
    percent difference, over subjects.
    """
    columns = ["label", "structure", "mean_volume_ml", "pearson_r", "mean_diff_pct", "mean_abs_diff_pct"]
    rows = []
    for (lab, name), g in results.groupby(["label", "structure"]):
        t, p = g["volume_true_ml"].to_numpy(), g["volume_pred_ml"].to_numpy()
        ok = t > 0
        if ok.sum() < 2:
            continue
        pct = 100 * (p[ok] - t[ok]) / t[ok]
        r = np.corrcoef(t[ok], p[ok])[0, 1] if t[ok].std() > 0 and p[ok].std() > 0 else np.nan
        rows.append({
            "label": lab, "structure": name, "mean_volume_ml": t[ok].mean(),
            "pearson_r": r, "mean_diff_pct": pct.mean(), "mean_abs_diff_pct": np.abs(pct).mean(),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("mean_volume_ml", ascending=False)
