# 02516 Assignment 3 – Part 1: Proposal Extraction

This repository contains the scaffolding for **Part 1** of the pothole detection project.  
The goal is to explore the dataset, extract object proposals, evaluate their quality, and
produce labelled proposals that can directly be used in the later stages (Parts 2 and 3)
where the actual detector will be trained.

## Repository layout

```
src/pothole_proposals/
    dataset.py         # Dataset discovery and Pascal VOC parsing utilities
    visualization.py   # Helpers for plotting images with bounding boxes
    proposals.py       # Selective Search wrapper (OpenCV + pure python fallback)
    evaluation.py      # IoU, recall curves and summary statistics
    labeling.py        # Assign class/background labels to each proposal
run_part1_pipeline.py  # CLI orchestration for the four required tasks
requirements.txt       # Python dependencies (install inside a virtualenv)
```

All outputs are written to `outputs/part1/` by default:

- `visualizations/` – images with the ground-truth boxes overlaid (Task 1).
- `proposals/<split>/` – JSON files with the raw proposals per image (Task 2).
- `evaluation/recall.json` – aggregated recall figures to determine how many proposals
  are required per image (Task 3).
- `training_proposals/<split>/` – proposals annotated as _pothole_ or _background_
  which form the input to the detector training code you will build in Part 2 (Task 4).

## Getting started

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run the pipeline. The script looks for the dataset under
   `/home/arian-sumak/Documents/DTU` by default; override `--dataset-root`
   if you keep it elsewhere.

   ```bash
   python run_part1_pipeline.py \
       --split train \
       --output-dir outputs/part1 \
       --visualization-count 6 \
       --proposal-limit 2000
   ```

The script performs all four tasks from the hand-out and prints an evaluation table that
helps you choose the minimum number of proposals needed to reach your desired recall.
The saved JSON files can be re-used by the notebooks or scripts you build for Parts 2–3.

## Notes

- The dataset loader expects Pascal VOC style XML files and a `splits.json` file that
  maps split names (`train`, `val`, `test`) to image identifiers. If `splits.json`
  is missing, every image will be treated as part of the selected split.
- All dataset access goes through the shared `dataloader/PotholeDataset.py` PyTorch
  loader so the training code in later parts can reuse the exact same data interface.
- OpenCV with `ximgproc` (from `opencv-contrib-python`) is used when available.
  Otherwise the pure-python `selectivesearch` implementation kicks in automatically.
- The code is intentionally modular so later parts can import `pothole_proposals`
  utilities without modifications.

Please refer to the inline documentation and comments in the sources for additional
details on each processing step.

For the test :

```
python test_pothole_dataset.py --max-images 3
```
