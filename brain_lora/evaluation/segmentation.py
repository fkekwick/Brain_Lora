"""End-to-end LoRA segmentation evaluation for the three BraTS cohorts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from brain_lora.config import get, require, require_dir, require_file
from brain_lora.data import BraTSSliceDataset, list_patients
from brain_lora.lora import assert_adapter_keys
from brain_lora.models import build_segmenter

from .metrics import RegionAccumulator, aggregate_patients


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@torch.no_grad()
def evaluate_segmentation(config: dict[str, Any]) -> dict[str, Any]:
    (model_name,) = require(config, "model.name")
    data_root = require_dir(config, "data.root")
    encoder_checkpoint = require_file(config, "model.checkpoint")
    adapter_checkpoint = require_file(config, "model.adapter_checkpoint")
    output_raw = get(config, "output.file")
    if output_raw in (None, ""):
        raise ValueError("Missing required configuration value: output.file")
    output_path = Path(output_raw).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = str(get(config, "data.dataset", "")).lower()
    if dataset_name not in {"gli", "ssa", "ped"}:
        raise ValueError("data.dataset must be one of: gli, ssa, ped")

    split_path = get(config, "data.split_file")
    if split_path:
        path = Path(split_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configured split file does not exist: {path}")
        with path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        split_name = str(get(config, "data.split", "test"))
        if split_name not in split:
            raise ValueError(f"Split JSON has no '{split_name}' array.")
        patient_ids = list(split[split_name])
    else:
        patient_ids = list_patients(data_root)

    device_name = str(get(config, "runtime.device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Set runtime.device: cpu.")
    device = torch.device(device_name)
    model = build_segmenter(
        str(model_name),
        encoder_checkpoint,
        rank=int(get(config, "lora.rank", 8)),
        lora_layers=str(get(config, "lora.layers", "all")),
        feature_layers=tuple(get(config, "model.feature_layers", [2, 5, 8, 11])),
        hidden_dim=int(get(config, "model.hidden_dim", 256)),
        output_size=int(get(config, "data.native_size", 240)),
    ).to(device)
    saved = torch.load(adapter_checkpoint, map_location=device, weights_only=False)
    state = saved.get("model", saved)
    assert_adapter_keys(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(trainable_names.intersection(missing))
    if missing_trainable or unexpected:
        raise RuntimeError(
            f"Adapter checkpoint mismatch: missing trainable={missing_trainable[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    model.eval()

    dataset = BraTSSliceDataset(
        data_root,
        patient_ids,
        get(config, "data.preprocessed_root"),
        get(config, "data.index_cache"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(get(config, "evaluation.batch_size", 16)),
        shuffle=False,
        num_workers=int(get(config, "runtime.num_workers", 8)),
    )
    accumulator = RegionAccumulator(
        {patient: dataset.volume_shape(patient) for patient in patient_ids}
    )
    for batch in loader:
        images = batch["image"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(images).argmax(1)
        accumulator.update(
            predictions.cpu().numpy().astype(np.uint8),
            batch["label"].numpy().astype(np.uint8),
            list(batch["patient"]),
            [int(value) for value in batch["slice"]],
        )
    rows = accumulator.rows(include_hd95=bool(get(config, "evaluation.hd95", True)))
    metrics = aggregate_patients(
        [{key: value for key, value in row.items() if key != "patient"} for row in rows]
    )
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "task": "segmentation",
        "lora_rank": int(get(config, "lora.rank", 8)),
        "lora_layers": str(get(config, "lora.layers", "all")),
        "n_patients": len(patient_ids),
        "checkpoint_file": adapter_checkpoint.name,
        "metrics": metrics,
        "per_patient": rows,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2)
    return result
