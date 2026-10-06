"""Frozen OASIS and ADNI linear-probe workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from brain_lora.config import get, output_dir, require, require_dir, require_file
from brain_lora.data.probes import load_labels, load_slice, subject_slice_paths
from brain_lora.lora import assert_adapter_keys
from brain_lora.models import build_segmenter

CLASSES = {
    "oasis": ("nondemented", "very_mild_dementia", "mild_dementia", "moderate_dementia"),
    "adni": ("CN", "MCI", "AD"),
}


def binary_probe_cohort(
    dataset: str, groups: list[str], known: tuple[str, ...]
) -> tuple[list[int], np.ndarray, tuple[str, str], list[str] | None]:
    """Build the historical dataset-specific binary probe cohort."""
    if dataset == "oasis":
        indices = list(range(len(groups)))
        target = np.array([0 if group == known[0] else 1 for group in groups])
        return indices, target, (known[0], "demented"), None
    if dataset == "adni":
        endpoints = (known[0], known[-1])
        indices = [index for index, group in enumerate(groups) if group in endpoints]
        target = np.array([1 if groups[index] == endpoints[1] else 0 for index in indices])
        excluded = [group for group in known if group not in endpoints]
        return indices, target, endpoints, excluded
    raise ValueError("Binary probe cohort is defined only for OASIS and ADNI.")


def _macro_auc(target: np.ndarray, probabilities: np.ndarray, classes: int) -> float | None:
    values = []
    for class_index in range(classes):
        binary = (target == class_index).astype(int)
        if binary.min() == binary.max():
            continue
        values.append(roc_auc_score(binary, probabilities[:, class_index]))
    return float(np.mean(values)) if values else None


def probe_arrays(
    features: np.ndarray,
    target: np.ndarray,
    classes: int,
    seeds: list[int],
    test_fraction: float = 0.3,
) -> dict[str, Any]:
    _, counts = np.unique(target, return_counts=True)
    stratify = target if counts.min() >= 2 else None
    rows = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            target,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify,
        )
        scaler = StandardScaler().fit(train_x)
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        classifier.fit(scaler.transform(train_x), train_y)
        transformed = scaler.transform(test_x)
        probability = classifier.predict_proba(transformed)
        prediction = classifier.predict(transformed)
        if classes == 2:
            auc = float(roc_auc_score(test_y, probability[:, 1]))
        else:
            auc = _macro_auc(test_y, probability, classes)
        rows.append(
            {
                "seed": seed,
                "auc": auc,
                "bacc": float(balanced_accuracy_score(test_y, prediction)),
            }
        )

    def aggregate(key: str) -> dict[str, float | None]:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    return {
        "auc": aggregate("auc"),
        "bacc": aggregate("bacc"),
        "per_seed": rows,
    }


@torch.no_grad()
def extract_subject_features(
    model: torch.nn.Module,
    root: Path,
    device: torch.device,
    batch_size: int,
    num_slices: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    subjects = sorted(
        path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("sub_")
    )
    for subject in subjects:
        slices = [load_slice(path) for path in subject_slice_paths(root, subject, num_slices)]
        tensors = [torch.from_numpy(image) for image in slices if image.max() > 0]
        if not tensors:
            continue
        images = torch.stack(tensors).unsqueeze(1)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = images.repeat(1, 3, 1, 1)
        vectors = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(device)
            batch = (batch - model.input_mean) / model.input_std
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                tokens = model.encoder.get_intermediate_layers(
                    batch, n=[11], reshape=False, norm=True
                )[0]
            vectors.append(tokens.float().mean(dim=1).cpu())
        output[subject] = torch.cat(vectors).mean(dim=0).numpy()
    return output


def run_linear_probe(config: dict[str, Any]) -> dict[str, Any]:
    (model_name,) = require(config, "model.name")
    root = require_dir(config, "data.root")
    labels_path = require_file(config, "data.labels_file")
    encoder_checkpoint = require_file(config, "model.checkpoint")
    destination = output_dir(config)
    dataset = str(get(config, "data.dataset", "")).lower()
    if dataset not in CLASSES:
        raise ValueError("data.dataset must be one of: oasis, adni")
    device_name = str(get(config, "runtime.device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Set runtime.device: cpu.")
    device = torch.device(device_name)
    model = build_segmenter(
        str(model_name),
        encoder_checkpoint,
        rank=int(get(config, "lora.rank", 8)),
        lora_layers=str(get(config, "lora.layers", "all")),
    )
    adapter_path = get(config, "model.adapter_checkpoint")
    adapted = adapter_path not in (None, "")
    if adapted:
        adapter = Path(adapter_path).expanduser()
        if not adapter.is_file():
            raise FileNotFoundError(f"Configured adapter checkpoint does not exist: {adapter}")
        saved = torch.load(adapter, map_location="cpu", weights_only=False)
        state = saved.get("model", saved)
        assert_adapter_keys(model, state)
        model.load_state_dict(state, strict=False)
    model.requires_grad_(False).to(device).eval()
    features = extract_subject_features(
        model,
        root,
        device,
        int(get(config, "probe.batch_size", 64)),
        int(get(config, "probe.num_slices", 0)),
    )
    labels = load_labels(labels_path)
    policy = get(config, "data.unlabeled_policy")
    if policy not in {"drop", "oasis_nondemented"}:
        raise ValueError("data.unlabeled_policy must be 'drop' or 'oasis_nondemented'.")
    if policy == "oasis_nondemented" and dataset != "oasis":
        raise ValueError("oasis_nondemented is valid only for the OASIS workflow.")
    if policy == "oasis_nondemented":
        subject_ids = sorted(features)
        groups = [labels.get(subject, "nondemented") for subject in subject_ids]
    else:
        subject_ids = sorted(subject for subject in features if subject in labels)
        groups = [labels[subject] for subject in subject_ids]
    if not subject_ids:
        raise RuntimeError("No labeled subjects produced features; check data roots and labels.")
    known = CLASSES[dataset]
    unknown = sorted(set(groups) - set(known))
    if unknown:
        raise ValueError(f"Unknown {dataset} label(s): {unknown}")
    matrix = np.stack([features[subject] for subject in subject_ids])
    seeds = [int(seed) for seed in get(config, "probe.seeds", [0, 1, 2])]

    multiclass_target = np.array([known.index(group) for group in groups])
    multiclass = probe_arrays(matrix, multiclass_target, len(known), seeds)
    multiclass.update(
        {
            "model": model_name,
            "adapted": adapted,
            "lora_rank": int(get(config, "lora.rank", 8)) if adapted else None,
            "lora_layers": str(get(config, "lora.layers", "all")) if adapted else None,
            "dataset": dataset,
            "task": "multiclass",
            "classes": list(known),
            "n_subjects": len(subject_ids),
        }
    )
    indices, binary_target, endpoints, excluded = binary_probe_cohort(dataset, groups, known)
    binary = probe_arrays(matrix[indices], binary_target, 2, seeds)
    binary.update(
        {
            "model": model_name,
            "adapted": adapted,
            "lora_rank": int(get(config, "lora.rank", 8)) if adapted else None,
            "lora_layers": str(get(config, "lora.layers", "all")) if adapted else None,
            "dataset": dataset,
            "task": "binary",
            "classes": list(endpoints),
            "excluded_classes": excluded,
            "n_subjects": len(indices),
        }
    )
    for task, result in (("multiclass", multiclass), ("binary", binary)):
        with (destination / f"{dataset}_{task}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return {"multiclass": multiclass, "binary": binary}
