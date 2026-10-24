"""Test-set evaluation at the original 256^3, 1 mm resolution.

The network works on 128^3 volumes. For evaluation its class scores are resized back
to the original grid before taking the argmax, so Dice and volumes are measured
against the original FreeSurfer label maps, not the downsampled ones.
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityRangePercentilesd

from .labels import NAMES, to_freesurfer, to_index
from .volumes import voxel_volume_ml


def _image_transform(size):
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Resized(keys="image", spatial_size=(size,) * 3, mode="area"),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ])


def predict(model, image_path, labels, size=128, device=None):
    """Segment one scan. Returns a FreeSurfer label map on the original voxel grid."""
    device = device or next(model.parameters()).device
    shape = nib.load(image_path).shape[:3]
    x = _image_transform(size)({"image": image_path})["image"]
    x = torch.as_tensor(np.asarray(x))[None].float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if device.type == "cuda":
            logits = logits.half()  # 45 x 256^3 scores fit on the GPU in half precision
        logits = F.interpolate(logits, size=shape, mode="trilinear", align_corners=False)
        index_map = logits.argmax(1)[0].cpu().numpy()
    return to_freesurfer(index_map, labels)


def dice_per_label(pred, truth, labels):
    """Dice for each label value (background excluded). NaN if a label is absent from both."""
    n = len(labels)
    p, t = to_index(pred, labels).ravel(), to_index(truth, labels).ravel()
    conf = np.bincount(t * n + p, minlength=n * n).reshape(n, n)
    out = {}
    for i, lab in enumerate(labels[1:], start=1):
        denom = conf[i, :].sum() + conf[:, i].sum()
        out[int(lab)] = 2 * conf[i, i] / denom if denom > 0 else np.nan
    return out


def evaluate(model, pairs, labels, size=128, save_dir=None, device=None):
    """Per-subject, per-structure Dice and volumes for a list of {'id','image','label'} dicts.

    If save_dir is given, each prediction is written there as <id>_pred.nii.gz with
    FreeSurfer label values and the original image's affine.
    """
    rows = []
    for k, pair in enumerate(pairs, start=1):
        pred = predict(model, pair["image"], labels, size, device)
        lab_img = nib.load(pair["label"])
        truth = np.asarray(lab_img.dataobj).astype(np.int64)
        # structures outside this label set count as background in the reference too
        truth = to_freesurfer(to_index(truth, labels), labels)
        vox_ml = voxel_volume_ml(lab_img)
        dice = dice_per_label(pred, truth, labels)
        for lab, d in dice.items():
            rows.append({
                "id": pair["id"], "label": lab, "structure": NAMES[lab], "dice": d,
                "volume_true_ml": (truth == lab).sum() * vox_ml,
                "volume_pred_ml": (pred == lab).sum() * vox_ml,
            })
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            img = nib.load(pair["image"])
            nib.save(nib.Nifti1Image(pred.astype(np.int16), img.affine, img.header),
                     os.path.join(save_dir, f"{pair['id']}_pred.nii.gz"))
        mean = np.nanmean(list(dice.values()))
        print(f"[{k}/{len(pairs)}] {pair['id']}  mean Dice {mean:.4f}")
    return pd.DataFrame(rows)


def summarise(results):
    """Mean and SD of Dice per structure over subjects, sorted from best to worst.

    Structures that are missing from every test scan (no Dice) are left out.
    """
    table = (results.groupby(["label", "structure"])["dice"]
             .agg(["mean", "std", "count"]).reset_index()
             .dropna(subset=["mean"])
             .sort_values("mean", ascending=False))
    return table.rename(columns={"count": "n_subjects"})
