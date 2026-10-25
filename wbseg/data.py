"""Finding scans, splitting subjects and building the MONAI transforms."""

import json
import os
import re
from glob import glob

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
)

from .labels import to_index


def subject_id(path):
    """OASIS-3 scan ID from a file name, e.g. OAS30001 or OAS30001_MR_d0129.

    Falls back to the file name without extension if there is no OASIS ID in it.
    """
    name = os.path.basename(path)
    match = re.search(r"OAS\d+(_MR_d\d+)?", name)
    return match.group(0) if match else name.split(".")[0]


def subject_of(scan_id):
    """Subject part of a scan ID (OAS30001_MR_d0129 -> OAS30001)."""
    return scan_id.split("_MR_")[0]


def find_pairs(image_dir, label_dir):
    """Match each T1 image to its aseg label map.

    Files are matched by OASIS ID. If the names carry no OASIS ID, the two sorted
    file lists are paired in order, which needs the same number of files in each folder.
    """
    images = sorted(glob(os.path.join(str(image_dir), "**", "*.nii*"), recursive=True))
    labels = sorted(glob(os.path.join(str(label_dir), "**", "*.nii*"), recursive=True))
    print(f"found {len(images)} images in {image_dir}")
    print(f"found {len(labels)} label maps in {label_dir}")
    if not images or not labels:
        print("check IMAGE_DIR and LABEL_DIR: one of the folders has no .nii / .nii.gz files")
        return []

    # 1) same scan ID (OAS30001_MR_d0129), 2) same subject (OAS30001), 3) sorted order
    pairs = []
    for key in (subject_id, lambda p: subject_of(subject_id(p))):
        label_by_id = {key(p): p for p in labels}
        pairs = [{"id": subject_id(i), "image": i, "label": label_by_id[key(i)]}
                 for i in images if key(i) in label_by_id]
        if pairs:
            break
    if not pairs and len(images) == len(labels):
        print("file names do not share an ID, pairing images and labels in sorted order")
        pairs = [{"id": subject_id(i), "image": i, "label": l} for i, l in zip(images, labels)]
    if len(pairs) < len(images):
        print(f"warning: {len(images) - len(pairs)} images have no matching label map")
    return pairs


def split_subjects(pairs, n_val, n_test, seed=0):
    """Random split by subject, so all scans of one person end up in the same set.

    n_val and n_test are numbers of subjects.
    """
    subjects = sorted({subject_of(p["id"]) for p in pairs})
    order = np.random.default_rng(seed).permutation(len(subjects))
    test = {subjects[i] for i in order[:n_test]}
    val = {subjects[i] for i in order[n_test:n_test + n_val]}
    split = {"train": [], "val": [], "test": []}
    for p in pairs:
        s = subject_of(p["id"])
        split["test" if s in test else "val" if s in val else "train"].append(p)
    return split


def save_split(split, path):
    with open(path, "w") as f:
        json.dump({k: [p["id"] for p in v] for k, v in split.items()}, f, indent=1)


def get_transforms(labels, size=128, train=False):
    """Load, map labels to class indices, resize the whole head to size^3 and scale intensity."""
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="label", func=lambda x: to_index(x, labels).astype(np.float32)),
        Resized(keys=["image", "label"], spatial_size=(size,) * 3, mode=("area", "nearest")),
        ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ]
    if train:
        # no left-right flips: they would swap left and right structures
        steps += [
            RandAffined(
                keys=["image", "label"], prob=0.5, mode=("bilinear", "nearest"),
                rotate_range=(0.15,) * 3, scale_range=(0.08,) * 3, padding_mode="zeros",
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            RandGaussianNoised(keys="image", std=0.02, prob=0.2),
        ]
    return Compose(steps)
