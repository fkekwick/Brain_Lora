"""LoRA segmentation training on a patient-level BraTS split."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from brain_lora.config import get, output_dir, require, require_dir, require_file
from brain_lora.data import BraTSSliceDataset
from brain_lora.evaluation.metrics import RegionAccumulator, aggregate_patients
from brain_lora.models import build_segmenter

from .losses import DiceCrossEntropyLoss


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    base_lr: float,
    minimum_lr: float,
) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (minimum_lr + (base_lr - minimum_lr) * cosine) / base_lr

    return LambdaLR(optimizer, scale)


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Set runtime.device: cpu.")
    return torch.device(name)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: BraTSSliceDataset,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float | int | None]]:
    model.eval()
    accumulator = RegionAccumulator(
        {patient: dataset.volume_shape(patient) for patient in dataset.patient_ids}
    )
    total_loss = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_function(logits, labels)
        total_loss += float(loss) * images.shape[0]
        count += images.shape[0]
        accumulator.update(
            logits.argmax(1).cpu().numpy().astype(np.uint8),
            labels.cpu().numpy().astype(np.uint8),
            list(batch["patient"]),
            [int(value) for value in batch["slice"]],
        )
    rows = accumulator.rows(include_hd95=False)
    numeric_rows = [{key: value for key, value in row.items() if key != "patient"} for row in rows]
    return total_loss / max(count, 1), aggregate_patients(numeric_rows)


def train_segmentation(config: dict[str, Any]) -> dict[str, Any]:
    (model_name,) = require(config, "model.name")
    data_root = require_dir(config, "data.root")
    split_path = require_file(config, "data.split_file")
    encoder_checkpoint = require_file(config, "model.checkpoint")
    destination = output_dir(config)
    with split_path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    for key in ("train", "val", "test"):
        if key not in split or not isinstance(split[key], list):
            raise ValueError(f"Split JSON requires an array named '{key}'.")

    seed = int(get(config, "training.seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device(str(get(config, "runtime.device", "cuda")))
    preprocessed = get(config, "data.preprocessed_root")
    index_cache = get(config, "data.index_cache")
    train_data = BraTSSliceDataset(data_root, split["train"], preprocessed, index_cache)
    val_data = BraTSSliceDataset(data_root, split["val"], preprocessed, index_cache)
    batch_size = int(get(config, "training.batch_size", 16))
    workers = int(get(config, "runtime.num_workers", 8))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = build_segmenter(
        str(model_name),
        encoder_checkpoint,
        rank=int(get(config, "lora.rank", 8)),
        lora_layers=str(get(config, "lora.layers", "all")),
        feature_layers=tuple(get(config, "model.feature_layers", [2, 5, 8, 11])),
        hidden_dim=int(get(config, "model.hidden_dim", 256)),
        output_size=int(get(config, "data.native_size", 240)),
    ).to(device)
    loss_function = DiceCrossEntropyLoss(
        ignore_background=bool(get(config, "training.ignore_background", False))
    )
    learning_rate = float(get(config, "training.learning_rate", 1e-3))
    minimum_lr = float(get(config, "training.minimum_learning_rate", 1e-5))
    epochs = int(get(config, "training.epochs", 50))
    gradient_accumulation = int(get(config, "training.gradient_accumulation_steps", 1))
    if gradient_accumulation <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=float(get(config, "training.weight_decay", 1e-2)),
    )
    scheduler = cosine_schedule(
        optimizer,
        max(1, (len(train_loader) // gradient_accumulation) * epochs),
        float(get(config, "training.warmup_fraction", 0.1)),
        learning_rate,
        minimum_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history_path = destination / "training.csv"
    best = -1.0
    start_epoch = 0
    resume_path = destination / "checkpoint.pth"
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved["best_validation_dice"])
        torch.set_rng_state(saved["torch_rng"])
        if saved.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])

    history_exists = history_path.is_file() and start_epoch > 0
    with history_path.open("a" if history_exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not history_exists:
            writer.writerow(
                (
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "dice_WT",
                    "dice_TC",
                    "dice_ET",
                    "dice_mean",
                )
            )
        for epoch in range(start_epoch, epochs):
            model.train()
            running = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(train_loader):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = loss_function(model(images), labels) / gradient_accumulation
                scaler.scale(loss).backward()
                if (batch_index + 1) % gradient_accumulation == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                running += float(loss) * gradient_accumulation * images.shape[0]
                seen += images.shape[0]
            validation_loss, metrics = evaluate_loader(
                model, val_loader, val_data, loss_function, device
            )
            writer.writerow(
                (
                    epoch,
                    running / max(seen, 1),
                    validation_loss,
                    metrics["dice_WT_mean"],
                    metrics["dice_TC_mean"],
                    metrics["dice_ET_mean"],
                    metrics["dice_mean"],
                )
            )
            checkpoint = {
                "model": model.trainable_state_dict(),
                "epoch": epoch,
                "val_dice_mean": metrics["dice_mean"],
            }
            torch.save(checkpoint, destination / "last_model.pt")
            score = float(metrics["dice_mean"] or 0.0)
            if score > best:
                best = score
                torch.save(checkpoint, destination / "best_model.pt")
            resume = {
                "model": model.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_validation_dice": best,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(),
            }
            temporary = destination / "checkpoint.pth.tmp"
            torch.save(resume, temporary)
            os.replace(temporary, resume_path)

    summary = {
        "model": model_name,
        "dataset": "brats_gli",
        "task": "segmentation",
        "lora_rank": int(get(config, "lora.rank", 8)),
        "lora_layers": str(get(config, "lora.layers", "all")),
        "seed": seed,
        "epochs": epochs,
        "gradient_accumulation_steps": gradient_accumulation,
        "best_validation_dice": best,
        "split_counts": {key: len(split[key]) for key in ("train", "val", "test")},
    }
    with (destination / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
