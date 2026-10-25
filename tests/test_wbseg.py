import json

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import DataLoader, Dataset

from wbseg.data import find_pairs, get_transforms, save_split, split_subjects, subject_id
from wbseg.evaluate import dice_per_label, evaluate, summarise
from wbseg.labels import FULL_LABELS, NAMES, REDUCED_LABELS, colour_table, to_freesurfer, to_index
from wbseg.train import build_model, train
from wbseg.volumes import agreement, label_volumes, volume_table


def test_label_sets():
    assert len(FULL_LABELS) == 45 and len(REDUCED_LABELS) == 28
    assert set(REDUCED_LABELS) <= set(FULL_LABELS)
    assert all(int(l) in NAMES for l in FULL_LABELS)


def test_mapping_round_trip():
    fs = np.array([[0, 2, 3], [255, 17, 53]])
    idx = to_index(fs, FULL_LABELS)
    assert idx.max() == 44
    assert np.array_equal(to_freesurfer(idx, FULL_LABELS), fs)


def test_unknown_and_dropped_labels_become_background():
    fs = np.array([0, 18, 72, 1035, 17])  # 72 and 1035 are not aseg labels used here
    assert to_index(fs, FULL_LABELS).tolist()[2:4] == [0, 0]
    red = to_index(fs, REDUCED_LABELS)
    assert red[1] == 0  # amygdala is not in the reduced set
    assert to_freesurfer(red, REDUCED_LABELS)[4] == 17


def test_dice():
    truth = np.zeros((10, 10, 10), int)
    truth[:5] = 17
    pred = np.zeros_like(truth)
    pred[:4] = 17
    d = dice_per_label(pred, truth, FULL_LABELS)
    assert d[17] == pytest.approx(2 * 400 / (500 + 400))
    assert np.isnan(d[53])


def test_colour_table_without_lut():
    t = colour_table(FULL_LABELS)
    assert t.shape == (45, 3) and t[0].sum() == 0


def test_subject_id():
    assert subject_id("/x/OAS30001_MR_d0129_T1w.nii.gz") == "OAS30001_MR_d0129"
    assert subject_id("/x/OAS30001_T1w.nii.gz") == "OAS30001"
    assert subject_id("/x/scan_7.nii.gz") == "scan_7"


def test_sessions_of_one_subject_stay_together():
    pairs = [{"id": f"OAS3000{s}_MR_d000{d}"} for s in range(6) for d in range(2)]
    split = split_subjects(pairs, n_val=1, n_test=2)
    assert [len(split[k]) for k in ("train", "val", "test")] == [6, 2, 4]
    subjects = [{p["id"][:8] for p in split[k]} for k in ("train", "val", "test")]
    assert not (subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2])


def test_pairing_when_names_differ(tmp_path):
    for d in ("img", "lab"):
        (tmp_path / d).mkdir()
    for s in ("OAS30001", "OAS30002"):
        for d, name in (("img", f"{s}_MR_d0129_T1w.nii.gz"), ("lab", f"{s}_aseg.nii.gz")):
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.eye(4)), tmp_path / d / name)
    pairs = find_pairs(tmp_path / "img", tmp_path / "lab")
    assert len(pairs) == 2 and all(p["label"].split("/")[-1][:8] == p["id"][:8] for p in pairs)


def test_empty_folder(tmp_path):
    assert find_pairs(tmp_path / "nothing", tmp_path / "nothing") == []


def test_pairing_without_oasis_ids(tmp_path):
    for d in ("img", "lab"):
        (tmp_path / d).mkdir()
    for i in range(3):
        for d, name in (("img", f"image_{i}.nii.gz"), ("lab", f"seg_{i}.nii.gz")):
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.eye(4)), tmp_path / d / name)
    pairs = find_pairs(tmp_path / "img", tmp_path / "lab")
    assert [p["label"].endswith(f"seg_{i}.nii.gz") for i, p in enumerate(pairs)] == [True] * 3


def _fake_subject(path_img, path_lab, rng, n=40):
    """A head-like volume: two hemispheres of 'white matter' and 'cortex' plus a few small structures."""
    z, y, x = np.mgrid[:n, :n, :n] - n / 2
    r = np.sqrt(x**2 + y**2 + z**2) + rng.normal(0, 0.5, (n, n, n))
    lab = np.zeros((n, n, n), np.int16)
    left = x < 0
    lab[(r < 16) & left], lab[(r < 16) & ~left] = 3, 42
    lab[(r < 12) & left], lab[(r < 12) & ~left] = 2, 41
    lab[(np.abs(x + 5) < 2) & (np.abs(y) < 2) & (np.abs(z) < 4)] = 17
    lab[(np.abs(x - 5) < 2) & (np.abs(y) < 2) & (np.abs(z) < 4)] = 53
    lab[(np.abs(x) < 1.5) & (np.abs(y + 5) < 2) & (np.abs(z) < 3)] = 4
    intensity = {0: 0, 2: 1.0, 41: 1.0, 3: 0.6, 42: 0.6, 17: 0.75, 53: 0.75, 4: 0.2}
    img = np.vectorize(intensity.get)(lab).astype(np.float32) + rng.normal(0, 0.03, lab.shape)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    nib.save(nib.Nifti1Image(img.astype(np.float32), affine), path_img)
    nib.save(nib.Nifti1Image(lab, affine), path_lab)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("oasis")
    (root / "img").mkdir()
    (root / "lab").mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        sid = f"OAS3{i:04d}"
        _fake_subject(root / "img" / f"{sid}_T1w.nii.gz", root / "lab" / f"{sid}_aseg.nii.gz", rng)
    return root


@pytest.mark.parametrize("label_set", ["full", "reduced"])
def test_pipeline_end_to_end(fake_data, tmp_path, label_set):
    torch.manual_seed(0)
    labels = FULL_LABELS if label_set == "full" else REDUCED_LABELS
    pairs = find_pairs(fake_data / "img", fake_data / "lab")
    assert len(pairs) == 8
    split = split_subjects(pairs, n_val=2, n_test=2)
    ids = [{p["id"] for p in split[k]} for k in ("train", "val", "test")]
    assert sum(map(len, ids)) == 8 and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    save_split(split, tmp_path / "split.json")
    assert len(json.load(open(tmp_path / "split.json"))["test"]) == 2

    size = 32
    train_ds = Dataset(split["train"], get_transforms(labels, size, train=True))
    val_ds = Dataset(split["val"], get_transforms(labels, size))
    batch = train_ds[0]
    assert tuple(batch["image"].shape) == (1, size, size, size)
    assert batch["label"].max() < len(labels)

    model = build_model(len(labels))
    out = tmp_path / "run"
    hist = train(model, DataLoader(train_ds, batch_size=2, shuffle=True), DataLoader(val_ds, batch_size=1),
                 len(labels), out, max_epochs=4, val_every=2, lr=3e-3, device=torch.device("cpu"))
    assert len(hist["train_loss"]) == 4 and len(hist["val_dice"]) == 2
    assert (out / "best_model.pt").exists()

    # resuming with a higher epoch limit continues from the checkpoint
    hist2 = train(build_model(len(labels)), DataLoader(train_ds, batch_size=2), DataLoader(val_ds, batch_size=1),
                  len(labels), out, max_epochs=6, val_every=2, device=torch.device("cpu"))
    assert hist2["epoch"] == [1, 2, 3, 4, 5, 6]

    model.load_state_dict(torch.load(out / "best_model.pt"))
    res = evaluate(model, split["test"], labels, size=size, save_dir=tmp_path / "pred", device=torch.device("cpu"))
    assert set(res["id"]) == {p["id"] for p in split["test"]}
    present = {2, 3, 4, 17, 41, 42, 53} & set(labels.tolist())
    assert set(summarise(res)["label"]) == present  # absent structures are left out

    # predictions keep the original grid and affine, and volumes use the real voxel size
    pred_file = tmp_path / "pred" / f"{split['test'][0]['id']}_pred.nii.gz"
    pred_img, lab_img = nib.load(pred_file), nib.load(split["test"][0]["label"])
    assert pred_img.shape == lab_img.shape and np.allclose(pred_img.affine, lab_img.affine)
    vols = label_volumes(split["test"][0]["label"], labels)
    expected = (np.asarray(lab_img.dataobj) == 2).sum() * 1.5**3 / 1000
    if 2 in labels:
        assert vols[2] == pytest.approx(expected)
    table = volume_table(tmp_path / "pred", labels)
    assert table["id"].str.startswith("OAS").all()
    assert {"structure", "pearson_r"} <= set(agreement(res).columns)
