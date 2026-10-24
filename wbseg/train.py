"""3D U-Net and the training loop (with checkpoints so a Colab run can be resumed)."""

import json
import os
import time

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.utils import one_hot


def build_model(n_classes):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def validate(model, loader, n_classes, device):
    """Mean Dice over all foreground classes, on the resized (training resolution) volumes."""
    metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).long()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            pred = one_hot(logits.argmax(1, keepdim=True), n_classes)
            metric(y_pred=pred, y=one_hot(y, n_classes))
    score = metric.aggregate().item()
    metric.reset()
    return score


def train(model, train_loader, val_loader, n_classes, out_dir, max_epochs=200,
          val_every=2, patience=15, lr=1e-3, device=None):
    """Train with DiceCE loss and Adam, keep the model with the best validation Dice.

    patience is counted in validation rounds. Everything needed to resume is in
    out_dir/checkpoint.pt, the best weights are in out_dir/best_model.pt and the
    loss and Dice curves are in out_dir/history.json.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    history = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": []}
    start, best, best_epoch, bad_rounds = 0, -1.0, 0, 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history, start = ckpt["history"], ckpt["epoch"]
        best, best_epoch, bad_rounds = ckpt["best"], ckpt["best_epoch"], ckpt["bad_rounds"]
        print(f"resuming after epoch {start} (best Dice {best:.4f} at epoch {best_epoch})")

    for epoch in range(start + 1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total, steps = 0.0, 0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(total / steps)
        msg = f"epoch {epoch}/{max_epochs}  loss {total / steps:.4f}"

        if epoch % val_every == 0:
            dice = validate(model, val_loader, n_classes, device)
            history["val_epoch"].append(epoch)
            history["val_dice"].append(dice)
            msg += f"  val Dice {dice:.4f}"
            if dice > best:
                best, best_epoch, bad_rounds = dice, epoch, 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
                msg += "  (best)"
            else:
                bad_rounds += 1

        print(f"{msg}  [{time.time() - t0:.0f}s]")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "epoch": epoch,
            "best": best, "best_epoch": best_epoch, "bad_rounds": bad_rounds,
        }, ckpt_path)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)

        if bad_rounds >= patience:
            print(f"stopping early: no improvement in {patience} validation rounds")
            break

    print(f"best validation Dice {best:.4f} at epoch {best_epoch}")
    return history
