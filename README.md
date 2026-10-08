# Brain LoRA

LoRA adaptation of DINOv3, SigLIP2, BioMedCLIP, and MerMED for BraTS brain-tumour
segmentation, plus frozen OASIS/ADNI linear probes and JSON-based result analysis.

## Environment

- Python 3.10 or newer
- PyTorch appropriate for the target CPU/GPU platform
- User-supplied upstream encoder checkpoints and legally accessible datasets

Install PyTorch first, then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

DINOv3 is downloaded through Torch Hub on its first use. The other encoders use
the checkpoint supplied in the run configuration.

## Training configuration

Start from `configs/workflows/train_brats_gli.yaml` and merge the desired model
template from `configs/models/`. Supply paths outside this repository:

```yaml
model:
  name: dinov3
  checkpoint: /path/to/encoder_checkpoint
data:
  root: /path/to/brats_gli
  preprocessed_root: null
  split_file: /path/to/patient_split.json
  index_cache: /path/to/valid_slice_index.json
  native_size: 240
lora:
  rank: 8
  layers: all
training:
  epochs: 50
  batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 0.001
  minimum_learning_rate: 0.00001
  weight_decay: 0.01
  warmup_fraction: 0.1
  seed: 42
  ignore_background: false
runtime:
  device: cuda
  num_workers: 8
output:
  dir: /path/to/output_run
```

The split JSON must contain patient-ID arrays named `train`, `val`, and `test`.
BraTS patient directories must provide the four modalities `t1n`, `t1c`, `t2w`,
`t2f`, and a segmentation file. Label value 4 is remapped to class 3.

Run training:

```bash
brain-lora train-seg --config /path/to/train.yaml
```

This writes `best_model.pt`, `last_model.pt`, resumable `checkpoint.pth`,
`training.csv`, and `run.json` to `output.dir`.

## Other commands

```bash
brain-lora eval-seg --config /path/to/eval.yaml
brain-lora probe --config /path/to/probe.yaml
brain-lora aggregate --config /path/to/aggregate.yaml
brain-lora paper-analysis --config /path/to/paper_analysis.yaml
```

Use the workflow templates under `configs/workflows/` as the field reference.
All input and output paths must be set explicitly; no dataset, split, weight,
checkpoint, feature cache, or result file is included in this repository.
