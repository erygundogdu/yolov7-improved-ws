# Sliced detection and evaluation

Run these commands from the repository root. Weights and data paths remain
relative to the working directory. Detection results default to `sahi/runs/exp`
(then `exp2`, `exp3`, etc.), regardless of the working directory. An explicit
`--project` overrides that location.

## Detection

```powershell
python sahi/detect_sahi.py --weights "best.pt" --source "images_12/images" --img-size 640 --slice-size 640
```

To draw slice windows, add `--draw-windows`:

```powershell
python sahi/detect_sahi.py --weights "best.pt" --source "images_12/images" --img-size 640 --slice-size 640 --draw-windows --name "windows"
```

This replaces `detectv1.py`: red rectangles show the actual inference windows,
and green boxes show merged detections with class IDs and confidence scores.
Drawing happens after inference and does not change detections or saved labels.
The overlay is included in saved images/videos and in `--view-img` displays.
Without `--draw-windows`, the existing detection visualization is used.

Set overlap with `--slice-overlap-h` and `--slice-overlap-w` (both default to 0.2).
The old `detectv1.py --slice-overlap VALUE` corresponds to setting both options.
Use `--project` and `--name` instead of its old `--output` option.
Detector defaults remain those of `detect_sahi.py`, including tracing and NMS;
use `--no-trace` if you need untraced inference.

## Evaluation

```powershell
python sahi/test_sahi.py --weights "best.pt" --data "data/coco.yaml" --sahi --batch-size 1
```

Replace `data/coco.yaml` with your detector dataset YAML. With `--sahi`, evaluation
uses its existing 1664-pixel canvas and nine 640-pixel windows at offsets 0, 512,
and 1024. Without `--sahi`, it runs ordinary full-image evaluation. Detection's
slice options do not configure this fixed evaluation grid.

Shared YOLO imports still come from the repository's `models/` and `utils/`.
The former root-level `detect_sahi.py`, `test_sahi.py`, and `detectv1.py` commands
are replaced by the two scripts here.
