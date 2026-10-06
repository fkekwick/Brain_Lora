"""Analysis over documented JSON outputs only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from brain_lora.config import get


def _paths(config: dict[str, Any], key: str) -> list[Path]:
    raw = get(config, key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Configuration value '{key}' must be a nonempty list of JSON files.")
    paths = [Path(item).expanduser() for item in raw]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing analysis input(s): {missing}")
    return paths


def _output(config: dict[str, Any]) -> Path:
    raw = get(config, "output.file")
    if raw in (None, ""):
        raise ValueError("Missing required configuration value: output.file")
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _metric(row: dict[str, Any], name: str) -> float | None:
    aliases = {"bacc": ("bacc", "balanced_accuracy"), "auc": ("auc",)}
    for key in aliases[name]:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _model_family(value: object) -> str:
    model = str(value)
    for suffix in ("_supervised", "_zeroshot"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def _segmentation_metrics(result: dict[str, Any]) -> dict[str, float | None] | None:
    if isinstance(result.get("metrics"), dict):
        return result["metrics"]
    dice = result.get("dice")
    if not isinstance(dice, dict):
        return None
    converted: dict[str, float | None] = {}
    for region in ("WT", "TC", "ET"):
        block = dice.get(region)
        if isinstance(block, dict):
            converted[f"dice_{region}_mean"] = block.get("mean")
            converted[f"dice_{region}_std"] = block.get("std")
    converted["dice_mean"] = dice.get("mean")
    hd95 = result.get("hd95")
    if isinstance(hd95, dict):
        for region in ("WT", "TC", "ET"):
            block = hd95.get(region)
            if isinstance(block, dict):
                converted[f"hd95_{region}_mean"] = block.get("mean")
                converted[f"hd95_{region}_std"] = block.get("std")
                converted[f"hd95_{region}_n_inf"] = block.get("n_inf", 0)
    return converted


def aggregate_results(config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in _paths(config, "analysis.inputs"):
        result = _read(path)
        key = (
            str(result.get("model", "unknown")),
            str(result.get("dataset", "unknown")),
            str(result.get("task", "segmentation")),
            result.get("adapted"),
            result.get("lora_rank"),
            result.get("lora_layers"),
        )
        grouped[key].append(result)
    summary: dict[str, Any] = {"groups": []}
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        model, dataset, task, adapted, rank, layers = key
        block: dict[str, Any] = {
            "model": model,
            "dataset": dataset,
            "task": task,
            "adapted": adapted,
            "lora_rank": rank,
            "lora_layers": layers,
            "n_files": len(rows),
        }
        if all("per_seed" in row for row in rows):
            seed_rows = [item for row in rows for item in row["per_seed"]]
            for metric in ("auc", "bacc"):
                values = [
                    value for item in seed_rows if (value := _metric(item, metric)) is not None
                ]
                block[metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
                    "n": len(values),
                }
        elif all(_segmentation_metrics(row) is not None for row in rows):
            converted_rows = [_segmentation_metrics(row) or {} for row in rows]
            keys = sorted(
                {key for metrics in converted_rows for key in metrics if key.endswith("_mean")}
            )
            block["metrics"] = {}
            for key in keys:
                values = [metrics.get(key) for metrics in converted_rows]
                values = [float(value) for value in values if value is not None]
                block["metrics"][key] = {
                    "mean": float(np.mean(values)) if values else None,
                    "seed_std": float(np.std(values)) if values else None,
                    "n": len(values),
                }
                if key.startswith("hd95_"):
                    prefix = key.removesuffix("_mean")
                    patient_stds = [metrics.get(f"{prefix}_std") for metrics in converted_rows]
                    patient_stds = [float(value) for value in patient_stds if value is not None]
                    infinity_counts = [
                        int(metrics.get(f"{prefix}_n_inf") or 0) for metrics in converted_rows
                    ]
                    block["metrics"][key].update(
                        {
                            "patient_mean": float(np.mean(values)) if values else None,
                            "patient_std": (float(np.mean(patient_stds)) if patient_stds else None),
                            "n_inf": sum(infinity_counts),
                            "n_seeds": len(values),
                        }
                    )
        else:
            raise ValueError(f"Mixed or unsupported result schema for {model}/{dataset}/{task}.")
        summary["groups"].append(block)
    with _output(config).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _seed_metrics(paths: list[Path]) -> dict[tuple[str, str, str, int], dict[str, float]]:
    output: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for path in paths:
        result = _read(path)
        model = _model_family(result.get("model"))
        dataset = str(result.get("dataset"))
        task = str(result.get("task"))
        if "per_seed" not in result:
            raise ValueError(f"Paper analysis expects a probe result with per_seed: {path}")
        for row in result["per_seed"]:
            output[(model, dataset, task, int(row["seed"]))] = row
    return output


def paper_analysis(config: dict[str, Any]) -> dict[str, Any]:
    frozen = _seed_metrics(_paths(config, "analysis.frozen_files"))
    adapted = _seed_metrics(_paths(config, "analysis.adapted_files"))
    common = sorted(set(frozen).intersection(adapted))
    if not common:
        raise ValueError("Frozen and adapted inputs have no matching dataset/task/seed rows.")
    comparisons = []
    groups = sorted({(model, dataset, task) for model, dataset, task, _ in common})
    for model, dataset, task in groups:
        seeds = [
            seed
            for current_model, ds, current_task, seed in common
            if (current_model, ds, current_task) == (model, dataset, task)
        ]
        for metric in ("auc", "bacc"):
            frozen_values = np.array(
                [_metric(frozen[(model, dataset, task, seed)], metric) for seed in seeds],
                dtype=float,
            )
            adapted_values = np.array(
                [_metric(adapted[(model, dataset, task, seed)], metric) for seed in seeds],
                dtype=float,
            )
            differences = adapted_values - frozen_values
            t_result = stats.ttest_rel(adapted_values, frozen_values)
            try:
                wilcoxon = float(stats.wilcoxon(adapted_values, frozen_values).pvalue)
            except ValueError:
                wilcoxon = None
            comparisons.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "seeds": seeds,
                    "frozen": frozen_values.tolist(),
                    "adapted": adapted_values.tolist(),
                    "mean_difference_adapted_minus_frozen": float(differences.mean()),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": wilcoxon,
                }
            )
    result = {
        "note": "Paired by dataset, task, and seed. Interpret tests cautiously for small n.",
        "comparisons": comparisons,
    }
    with _output(config).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
