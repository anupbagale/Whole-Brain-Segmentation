# Whole-brain segmentation of T1 MRI into 44 FreeSurfer structures

[![tests](https://github.com/anupbagale/Whole-Brain-Segmentation/actions/workflows/tests.yml/badge.svg)](https://github.com/anupbagale/Whole-Brain-Segmentation/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anupbagale/Whole-Brain-Segmentation/blob/main/notebooks/02_train_and_evaluate.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

FreeSurfer takes hours to segment one brain scan. A 3D U-Net can do it in seconds once it is trained. In this project I train a U-Net on OASIS-3 T1 scans to reproduce the full FreeSurfer `aseg` segmentation, all 44 structures at once, from the cerebral white matter down to the optic chiasm and the five parts of the corpus callosum.

![FreeSurfer labels and U-Net prediction on a test subject](figures/example_segmentation.png)

## Questions

1. **How well does one network segment all 44 structures?** Large structures are easy. Small ones like the amygdala, the accumbens or the vessel labels are only a few hundred voxels, which makes them hard to learn next to the white matter.
2. **Does it help to leave the small structures out?** I trained a second network on a reduced set of 27 structures and compared both on the structures they have in common.
3. **Are the volumes right?** Segmentations are often used to measure structure volumes, so I compared the U-Net volumes with the FreeSurfer volumes on the test subjects.

## Results

<!-- Fill in from results/summary.md after running notebook 02, then delete this line. -->

Test set of [n] subjects, Dice measured at the original 1 mm resolution against the FreeSurfer labels. Mean ± SD over test subjects.

| Model | Mean Dice, all its structures | Mean Dice, 27 shared structures |
|---|---|---|
| 45 classes (44 structures) | [x.xxx ± x.xxx] | [x.xxx ± x.xxx] |
| 28 classes (27 structures) | [x.xxx ± x.xxx] | [x.xxx ± x.xxx] |

[One or two sentences: which model is better on the shared structures, the Wilcoxon p value, and what that means.]

![Dice per structure on the test set](figures/dice_all_structures.png)

[One or two sentences on which structures work well and which do not.]

**Volumes.** [One or two sentences from results/volume_agreement.csv, for example the correlation and mean difference for the hippocampus and the lateral ventricles.]

![Predicted vs FreeSurfer volumes](figures/volume_agreement.png)

## How it works

- **Data.** T1-weighted scans from OASIS-3 with their FreeSurfer `aseg` label maps (256 × 256 × 256, 1 mm). Subjects are split at random into training, validation and test sets, and all scans of one person stay in the same set. The split is saved in `results/data_split.json`.
- **Labels.** FreeSurfer label values (2, 3, 4, ... 255) are mapped to class indices 0 to 44 for training and mapped back when predictions are saved. The reduced set keeps 27 structures and sets the rest to background. Both lists are in [`wbseg/labels.py`](wbseg/labels.py).
- **Preprocessing.** The whole head is resized to 128³ and intensities are scaled to 0 to 1 using the 0.5th and 99.5th percentiles. Training uses small random rotations and scaling, intensity scaling and noise. There are no left-right flips, because a flip would turn left structures into right ones.
- **Model.** MONAI 3D U-Net with residual units, 5 levels (16 to 256 channels), batch norm. Loss: Dice + cross-entropy. Adam, learning rate 1e-3 with cosine decay, mixed precision, early stopping on validation Dice.
- **Evaluation.** The network's class scores are resized back to 256³ before the argmax, so Dice and volumes are measured on the original grid against the original label maps. Predictions are saved as NIfTI files with FreeSurfer label values and the original image header, so they open directly in FreeView or ITK-SNAP.

## Notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| [01_data_and_labels](notebooks/01_data_and_labels.ipynb) | Checks the data, compares the two label sets, plots how big each structure is | CPU |
| [02_train_and_evaluate](notebooks/02_train_and_evaluate.ipynb) | Trains both U-Nets, tests them, writes `results/summary.md` | GPU |
| [03_structure_volumes](notebooks/03_structure_volumes.ipynb) | Compares U-Net and FreeSurfer volumes, exports a volume table | CPU |

Each notebook opens in Colab with the badge at the top of it. Training saves a checkpoint to Google Drive after every epoch, so if Colab disconnects you can run all cells again and it continues where it stopped.

## Running it

1. Get access to OASIS-3 (see below) and put the T1 images and the `aseg` label maps in two folders on your Google Drive.
2. Optionally add `FreeSurferColorLUT.txt` from a FreeSurfer installation to get the standard FreeSurfer colours in the figures.
3. Open a notebook in Colab, set the paths in the second code cell, and run all cells.

To run locally instead:

```bash
git clone https://github.com/anupbagale/Whole-Brain-Segmentation.git
cd Whole-Brain-Segmentation
pip install -r requirements.txt
python -m pytest tests     # about 15 seconds on CPU, uses small synthetic volumes
```

## Code

```
wbseg/
  labels.py     the 45- and 28-class label sets, structure names, label mapping, colours
  data.py       matching images to label maps, subject-level split, MONAI transforms
  train.py      U-Net and training loop with checkpoints and early stopping
  evaluate.py   prediction at the original resolution, Dice per structure
  volumes.py    structure volumes from any label map, agreement with FreeSurfer
  plots.py      figures
notebooks/      the three notebooks above
tests/          tests on synthetic brains, run by GitHub Actions on every push
```

## Limitations

- FreeSurfer's output is used as the ground truth, so the network learns FreeSurfer's mistakes too. Dice here measures agreement with FreeSurfer, not with manual tracing.
- Resizing the head to 128³ halves the resolution. This hurts thin and small structures most, which shows up in their Dice.
- All scans come from one dataset, so this does not show how the model would do on scans from other scanners or protocols.

## Data

No OASIS-3 images or labels are included here. They are available after accepting the OASIS data use agreement at [oasis-brains.org](https://www.oasis-brains.org).

Data were provided by OASIS-3: Longitudinal Multimodal Neuroimaging: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352, AW00006993.

## License

MIT for the code. The OASIS-3 data has its own terms.
