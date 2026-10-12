# Two-stage detection

| Script | Purpose |
| --- | --- |
| `prepare_class_data.py` | Extract 160-pixel crops from class-0 YOLO boxes |
| `generate_dataset.py` | Create train/val/test splits and augment training images |
| `classifier.py` | Train the binary classifier |
| `export_onnx_classifier.py` | Export `yeni.pt` to `classifier.onnx` |
| `detect_two_stage.py` | Run detection, crop classification, and Grad-CAM |

## Create a dataset

First collect source images and their YOLO bounding-box annotations:

```text
input_dataset/
  images/
    example.jpg
  labels/
    example.txt
```

Each label file must match its image's filename stem. Each annotation row is
`class_id x_center y_center width height`, with coordinates normalized to 0-1.
The crop script extracts only class-0 boxes. Include examples that can be labelled
as both `target` and `non_target`, with at least three independent source images
per binary class to create all three splits. Annotate source images first if no
YOLO label files exist. This script does not generate bounding-box annotations.

Then run from the repository root:

```powershell
python detect_two_stage/prepare_class_data.py --input "C:\Users\statb\Desktop\yolov7\images_12"
```

Input must contain `images/` and `labels/`, or be an images directory with a sibling
`labels/` directory. Without `--input`, the script prompts for the path. `--labels`
overrides the labels directory. Crops go to `detect_two_stage/crops/` by default.

Sort crops into `target/` and `non_target/` subfolders. YOLO class 0 does not identify
which binary class a crop belongs to. If every crop from an input belongs to one
binary class, use `--class-name target` or `--class-name non_target` to save directly
to that class subfolder.

```powershell
python detect_two_stage/generate_dataset.py --input "detect_two_stage/crops" --augment 2
```

You can also use existing class folders directly:

```powershell
python detect_two_stage/generate_dataset.py --target "C:/data/target" --non-target "C:/data/non_target" --augment 2
```

Output defaults to this script's directory regardless of the working directory:

```text
detect_two_stage/
  crops/
    target/
    non_target/
  dataset/
    train/
      target/
      non_target/
    val/
      target/
      non_target/
    test/
      target/
      non_target/
```

Default split: 80/10/10, seed 42. Override with `--split 0.7 0.15 0.15` or
`--seed 123`. Crops sharing the preparation script's source filename stay in one
split, including across classes. Preserve crop filenames for grouping. Each class
needs at least three independent source images. Small datasets have at least one
source per split, so actual proportions may differ. Arbitrarily renamed duplicates
and related video frames cannot be identified automatically.

`--augment 2` adds two resized/flipped copies per training image. Validation and
test images are copied without augmentation. Source files are retained. Nonempty
output folders are rejected. Use `--output` to select a new crop directory or
`--out` to select a new dataset directory.

## Single-image augmentation

```powershell
python detect_two_stage/generate_dataset.py --image "C:/data/target.jpg" --n 100
```

This preserves the original single-image workflow and defaults to
`detect_two_stage/generated_dataset/target/`. It does not create splits. Use
independent source images for splitting; augmented copies of one image should not
be distributed across training and evaluation sets.

## Training, export, and inference

```powershell
python detect_two_stage/classifier.py
python detect_two_stage/export_onnx_classifier.py
python detect_two_stage/detect_two_stage.py --det-weights "best.pt" --source "images_12/images"
```

Training reads `detect_two_stage/dataset/{train,val,test}`. For custom dataset
output, update `DATASET_DIR` in `classifier.py`. Training initializes from
`resnet18_imagenet.pth` and writes `detect_two_stage/yeni.pt`. Its existing
loop selects checkpoints using training accuracy; validation and test data are
loaded but not evaluated. Export reads `detect_two_stage/yeni.pt` and writes
`detect_two_stage/classifier.onnx`. Inference defaults to the same classifier
checkpoint; `--cls-weights` overrides it. These default artifact paths are anchored
to the scripts' directory, regardless of the working directory.

Inference saves results under `runs/detect_two_stage/exp` by default. Shared YOLO
code remains in `models/` and `utils/`. Inference requires `pytorch_grad_cam` from
the `grad-cam` package.

Run the scripts directly from `detect_two_stage/`; the old root-level wrappers
have been removed. `images_12/generate_dataset.py` still forwards to the generator
here.
