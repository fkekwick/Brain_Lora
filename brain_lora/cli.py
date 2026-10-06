"""Command-line entry points for all public workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from brain_lora.analysis import aggregate_results, paper_analysis
from brain_lora.config import load_config
from brain_lora.evaluation.probes import run_linear_probe
from brain_lora.evaluation.segmentation import evaluate_segmentation
from brain_lora.training import train_segmentation


def _configured(
    subparsers: argparse._SubParsersAction, name: str, help_text: str, function: Callable
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.set_defaults(function=function)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-lora")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _configured(subparsers, "train-seg", "train LoRA segmentation", train_segmentation)
    _configured(subparsers, "eval-seg", "evaluate BraTS segmentation", evaluate_segmentation)
    _configured(subparsers, "probe", "run a frozen linear probe", run_linear_probe)
    _configured(subparsers, "aggregate", "aggregate JSON result files", aggregate_results)
    _configured(subparsers, "paper-analysis", "run paired paper statistics", paper_analysis)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    config: dict[str, Any] = load_config(arguments.config) if arguments.config else {}
    result = arguments.function(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
