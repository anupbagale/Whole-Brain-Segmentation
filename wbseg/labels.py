"""FreeSurfer aseg label sets used in this project.

FULL_LABELS has the 44 aseg structures found in OASIS-3 plus background (45 classes).
REDUCED_LABELS keeps the 27 larger structures plus background (28 classes).
The network is trained on class indices 0..N-1, so labels are mapped to indices
before training and mapped back to FreeSurfer values when predictions are saved.
"""

import numpy as np

FULL_LABELS = np.array([
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31,
    41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63,
    77, 80, 85, 251, 252, 253, 254, 255,
])

REDUCED_LABELS = np.array([
    0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 28, 31,
    41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 60, 63,
])

LABEL_SETS = {"full": FULL_LABELS, "reduced": REDUCED_LABELS}

NAMES = {
    0: "Background",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    77: "WM-hypointensities",
    80: "non-WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
}


def to_index(label_map, labels):
    """FreeSurfer values -> class indices. Values not in `labels` become background."""
    label_map = np.asarray(label_map).astype(np.int64)
    lookup = np.zeros(max(labels.max(), label_map.max(initial=0)) + 1, dtype=np.int64)
    lookup[labels] = np.arange(len(labels))
    out = np.zeros_like(label_map)
    ok = label_map >= 0
    out[ok] = lookup[label_map[ok]]
    return out


def to_freesurfer(index_map, labels):
    """Class indices -> FreeSurfer values."""
    return labels[np.asarray(index_map).astype(np.int64)]


def read_lut(lut_path):
    """Read FreeSurferColorLUT.txt into {label: (r, g, b)} with values in 0..1."""
    colours = {}
    with open(lut_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            colours[int(parts[0])] = tuple(int(v) / 255 for v in parts[2:5])
    return colours


def colour_table(labels, lut_path=None):
    """(N, 3) array of colours for class indices 0..N-1.

    Uses the FreeSurfer colours if a lookup table is given, otherwise a fixed random palette.
    """
    if lut_path is not None:
        lut = read_lut(lut_path)
        table = np.array([lut.get(int(l), (0.5, 0.5, 0.5)) for l in labels])
    else:
        table = np.random.default_rng(1).uniform(0.15, 1.0, size=(len(labels), 3))
    table[0] = 0
    return table
