"""Figures used in the notebooks and the README."""

import matplotlib.pyplot as plt
import numpy as np

from .labels import to_index

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def overlay(image, label_fs, labels, colours, axis=0, index=None, alpha=0.45, ax=None, box=None):
    """Show one slice of a T1 image with its label map on top.

    box: optional (r0, r1, c0, c1) crop of the rotated slice, e.g. from brain_box().
    """
    index = image.shape[axis] // 2 if index is None else index
    img = np.rot90(np.take(image, index, axis=axis))
    lab = np.rot90(np.take(to_index(label_fs, labels), index, axis=axis))
    rgba = np.zeros(lab.shape + (4,))
    rgba[..., :3] = colours[lab]
    rgba[..., 3] = (lab > 0) * alpha
    if box is not None:
        r0, r1, c0, c1 = box
        img, rgba = img[r0:r1, c0:c1], rgba[r0:r1, c0:c1]
    ax = ax or plt.gca()
    ax.imshow(img, cmap="gray")
    ax.imshow(rgba)
    ax.axis("off")
    return ax


def brain_box(label_fs, axis, index, margin=6):
    """Crop box around the labelled brain in one slice, so figures show the brain and not the face."""
    lab = np.rot90(np.take(np.asarray(label_fs) > 0, index, axis=axis))
    rows, cols = np.where(lab)
    if len(rows) == 0:
        return None
    return (max(rows.min() - margin, 0), rows.max() + margin + 1,
            max(cols.min() - margin, 0), cols.max() + margin + 1)


def comparison_figure(image, truth, pred, labels, colours, index=None, axes=(0, 1, 2)):
    """Rows: FreeSurfer and prediction. Columns: sagittal, coronal, axial slices."""
    fig, grid = plt.subplots(2, len(axes), figsize=(3.2 * len(axes), 6.6))
    for col, axis in enumerate(axes):
        idx = image.shape[axis] // 2 if index is None else index[col]
        box = brain_box(truth, axis, idx)
        overlay(image, truth, labels, colours, axis=axis, index=idx, ax=grid[0, col], box=box)
        overlay(image, pred, labels, colours, axis=axis, index=idx, ax=grid[1, col], box=box)
    grid[0, 0].set_title("FreeSurfer", loc="left")
    grid[1, 0].set_title("3D U-Net", loc="left")
    fig.tight_layout()
    return fig


def training_curves(histories):
    """histories: {name: history dict from train()}."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for (name, h), c in zip(histories.items(), (BLUE, ORANGE)):
        a.plot(h["epoch"], h["train_loss"], color=c, lw=2, label=name)
        b.plot(h["val_epoch"], h["val_dice"], color=c, lw=2, label=name)
    a.set_title("Training loss (Dice + cross-entropy)", loc="left")
    b.set_title("Validation mean Dice", loc="left")
    for ax in (a, b):
        ax.set_xlabel("epoch")
        _clean(ax)
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def dice_by_structure(tables, names=("45 classes", "28 classes")):
    """Horizontal dot plot of mean test Dice per structure for one or two models.

    tables: list of summarise() outputs. Structures are ordered by the first table.
    """
    order = tables[0].sort_values("mean")["structure"].tolist()
    fig, ax = plt.subplots(figsize=(7, 0.24 * len(order) + 1.2))
    y = np.arange(len(order))
    for t, name, c, off in zip(tables, names, (BLUE, ORANGE), (0.15, -0.15)):
        t = t.set_index("structure").reindex(order)
        ax.errorbar(t["mean"], y + off, xerr=t["std"], fmt="o", ms=5, color=c,
                    ecolor=c, elinewidth=1, alpha=0.9, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Test Dice (mean ± SD over subjects)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(tables) > 1:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def volume_scatter(results, structures):
    """Predicted vs FreeSurfer volume for a few structures, one panel each."""
    fig, axes = plt.subplots(1, len(structures), figsize=(3.4 * len(structures), 3.4))
    for ax, name in zip(np.atleast_1d(axes), structures):
        g = results[results["structure"] == name]
        t, p = g["volume_true_ml"], g["volume_pred_ml"]
        lo, hi = min(t.min(), p.min()) * 0.95, max(t.max(), p.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
        ax.scatter(t, p, s=22, color=BLUE, edgecolor="white", lw=0.6)
        ok = len(g) > 2 and t.std() > 0 and p.std() > 0
        ax.set_title(f"{name}\nr = {np.corrcoef(t, p)[0, 1]:.2f}" if ok else name, loc="left", fontsize=9)
        ax.set_xlabel("FreeSurfer (ml)")
        ax.set_ylabel("U-Net (ml)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _clean(ax)
    fig.tight_layout()
    return fig


def bland_altman(results, structure):
    g = results[results["structure"] == structure]
    mean = (g["volume_true_ml"] + g["volume_pred_ml"]) / 2
    diff = g["volume_pred_ml"] - g["volume_true_ml"]
    md, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(mean, diff, s=22, color=BLUE, edgecolor="white", lw=0.6)
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax.axhline(v, color=GREY, lw=1, ls=ls)
    ax.set_title(f"{structure}: U-Net minus FreeSurfer", loc="left", fontsize=10)
    ax.set_xlabel("Mean of the two volumes (ml)")
    ax.set_ylabel("Difference (ml)")
    _clean(ax)
    fig.tight_layout()
    return fig
