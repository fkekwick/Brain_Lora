"""Patient-level BraTS region metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

REGIONS = ("WT", "TC", "ET")


def segmentation_regions(segmentation: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "WT": segmentation >= 1,
        "TC": np.isin(segmentation, (1, 3)),
        "ET": segmentation == 3,
    }


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.sum() == 0 and target.sum() == 0:
        return 0.0
    if prediction.sum() == 0 or target.sum() == 0:
        return float("inf")
    prediction_border = prediction ^ (distance_transform_edt(prediction) > 1)
    target_border = target ^ (distance_transform_edt(target) > 1)
    prediction_distance = distance_transform_edt(~prediction_border)
    target_distance = distance_transform_edt(~target_border)
    distances = np.concatenate(
        (target_distance[prediction_border], prediction_distance[target_border])
    )
    return float(np.percentile(distances, 95)) if distances.size else float("inf")


def evaluate_patient(
    prediction: np.ndarray, target: np.ndarray, include_hd95: bool = True
) -> dict[str, float]:
    prediction_regions = segmentation_regions(prediction)
    target_regions = segmentation_regions(target)
    result: dict[str, float] = {}
    for region in REGIONS:
        result[f"dice_{region}"] = dice_score(prediction_regions[region], target_regions[region])
        if include_hd95:
            result[f"hd95_{region}"] = hd95(prediction_regions[region], target_regions[region])
    return result


def aggregate_patients(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for region in REGIONS:
        dice_values = [row[f"dice_{region}"] for row in rows]
        result[f"dice_{region}_mean"] = float(np.mean(dice_values)) if dice_values else None
        result[f"dice_{region}_std"] = float(np.std(dice_values)) if dice_values else None
        key = f"hd95_{region}"
        hd_values = [row[key] for row in rows if key in row]
        if hd_values:
            finite = [value for value in hd_values if np.isfinite(value)]
            result[f"hd95_{region}_mean"] = float(np.mean(finite)) if finite else None
            result[f"hd95_{region}_std"] = float(np.std(finite)) if finite else None
            result[f"hd95_{region}_n_inf"] = len(hd_values) - len(finite)
    dice_means = [result[f"dice_{region}_mean"] for region in REGIONS]
    result["dice_mean"] = float(np.mean(dice_means)) if rows else None
    return result


class RegionAccumulator:
    """Accumulate slice-level predictions into patient volumes."""

    def __init__(self, shapes: dict[str, tuple[int, int, int]]) -> None:
        self.predictions = {key: np.zeros(shape, dtype=np.uint8) for key, shape in shapes.items()}
        self.targets = {key: np.zeros(shape, dtype=np.uint8) for key, shape in shapes.items()}

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        patients: list[str],
        slices: list[int],
    ) -> None:
        for prediction, target, patient, axial in zip(
            predictions, targets, patients, slices, strict=True
        ):
            self.predictions[patient][:, :, axial] = prediction
            self.targets[patient][:, :, axial] = target

    def rows(self, include_hd95: bool = True) -> list[dict[str, float | str]]:
        output: list[dict[str, float | str]] = []
        for patient in sorted(self.predictions):
            row: dict[str, float | str] = {"patient": patient}
            row.update(
                evaluate_patient(self.predictions[patient], self.targets[patient], include_hd95)
            )
            output.append(row)
        return output
