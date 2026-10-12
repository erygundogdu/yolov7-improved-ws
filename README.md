# YOLOv7 Improved Workspace

This repository extends YOLOv7 with NWD box loss, knowledge distillation,
BatchNorm L1 sparsity regularization, sliced inference, and two-stage detection.
The original YOLOv7 documentation and attribution are retained below. Its reported
benchmark results are upstream results, not measurements of these modifications.

## Added workflows

| Feature | Implementation | Configuration |
| --- | --- | --- |
| NWD + CIoU box loss | [utils/loss.py](utils/loss.py), [utils/general.py](utils/general.py) | Enabled in `ComputeLoss` and `ComputeLossOTA` |
| Knowledge distillation | [train.py](train.py), [kd_train.py](kd_train.py) | `--kd_flag --teacher-weights PATH` |
| BN L1 sparsity | [train.py](train.py) | `--bn-l1 --bn-lambda 0.001` |
| SAHI-style sliced inference | [sahi/detect_sahi.py](sahi/detect_sahi.py) | `--slice-size`, overlap options, `--draw-windows` |
| Sliced evaluation | [sahi/test_sahi.py](sahi/test_sahi.py) | `--sahi` |
| Detector + binary classifier | [detect_two_stage/](detect_two_stage/) | Crop preparation, dataset generation, training, export, inference |

Run the commands below from the repository root. Replace example dataset and
checkpoint paths with your own. Detector dataset YAML files describe image paths,
class count, and class names; the images and labels must be supplied separately.

### NWD loss

The box-regression loss in both `ComputeLoss` and `ComputeLossOTA` combines
Normalized Wasserstein Distance (NWD) similarity with Complete IoU (CIoU):

```text
L_box = mean(0.65 * (1 - NWD) + 0.35 * (1 - CIoU))
```

`bbox_nwd()` compares box centers and half-width/half-height differences, then
computes `exp(-distance / 6.0)`. Boxes are passed in feature-grid coordinates.
The coefficients and normalization constant are currently defined in the code;
there is no NWD command-line switch. `loss_ota: 0` in the hyperparameter YAML
selects `ComputeLoss`; `loss_ota: 1` selects `ComputeLossOTA`. Both include NWD.
The standard path retains CIoU-based objectness targets, while the OTA path uses
`0.4 + 0.6 * NWD` for matched objectness targets. Other loss classes are not
covered by this NWD modification.

### Knowledge distillation

```powershell
python train.py --weights "student.pt" --data "data/your_dataset.yaml" --hyp "data/hyp.scratch.p5.yaml" --img-size 640 640 --device 0 --kd_flag --teacher-weights "teacher.pt" --name "kd_training"
```

The teacher is loaded only when `--kd_flag` is enabled. Its checkpoint must contain
a `model` entry. Teacher parameters are frozen, the teacher runs in evaluation
mode without gradients, and only the student is optimized.

`kd_train.py` compares corresponding raw detection-head outputs using objectness
and classification BCE losses plus a Smooth L1 box loss. Box and class terms use
positions where teacher objectness exceeds 0.20. The training objective is:

```text
L = 0.30 * L_YOLO + 0.70 * L_KD
```

Teacher and student must have compatible detection-head tensor shapes, anchor/grid
layouts, and class ordering; this implementation does not align mismatched heads.
The KD helper currently allocates its default loss weights on CUDA device 0, so
the example uses `--device 0`; CPU and multi-device KD need device-handling changes.
The teacher path is saved in the run options. On resume, an explicit
`--teacher-weights` overrides the saved path.

### BN L1 sparsity regularization

```powershell
python train.py --weights "student.pt" --data "data/your_dataset.yaml" --hyp "data/hyp.scratch.p5.yaml" --bn-l1 --bn-lambda 0.001 --name "bn_sparse"
```

`--bn-l1` adds `bn_lambda * sum(abs(gamma))` over BatchNorm2d and SyncBatchNorm scale
weights. It is disabled by default; the default coefficient is `0.001`.
This encourages small BN scales but does not itself remove channels or produce
a pruned model. Scale parameters must remain trainable for the penalty to update
them. The coefficient must be finite and nonnegative.

BN L1 can also be combined with KD:

```powershell
python train.py --weights "student.pt" --data "data/your_dataset.yaml" --device 0 --kd_flag --teacher-weights "teacher.pt" --bn-l1 --bn-lambda 0.001 --name "kd_bn_sparse"
```

The combined objective is `0.30 * L_YOLO + 0.70 * L_KD + bn_lambda * sum(abs(gamma))`.

### SAHI-style sliced detection and evaluation

Detection runs the model on overlapping image crops, maps detections back to the
original image, and merges them with NMS. This is implemented directly in the repo
and does not require the external SAHI package.

```powershell
python sahi/detect_sahi.py --weights "best_vehicle_640.pt" --source "path/to/images" --img-size 640 --slice-size 640 --slice-overlap-h 0.2 --slice-overlap-w 0.2 --draw-windows
```

`--draw-windows` overlays red crop boundaries and green merged detections, replacing
the former `detectv1.py` debugging script. Omit the flag for normal visualization.
Detection outputs default to `sahi/runs/exp`, incrementing for subsequent runs.
`--project` and `--name` override the output location.

```powershell
python sahi/test_sahi.py --weights "best_vehicle_640.pt" --data "data/your_dataset.yaml" --sahi --batch-size 1
```

Evaluation's `--sahi` mode uses a fixed 1664 x 1664 canvas with nine 640 x 640
windows at offsets 0, 512, and 1024. It does not inherit the detector's configurable
slice settings. Without `--sahi`, evaluation uses ordinary full-image inference.
See [sahi/README.md](sahi/README.md) for details.

### Two-stage detection

The first stage proposes boxes with YOLOv7. The second stage expands each box,
resizes the crop to 160 x 160, and applies a ResNet18-based binary classifier with
multi-scale feature fusion. Inference annotates each detection with a `yes`/`no`
decision at sigmoid probability 0.5 and saves classifier Grad-CAM visualizations.
Negative classifications are annotated rather than removed from the output.

Start with an input directory containing `images/` and matching YOLO `labels/`:

```powershell
python detect_two_stage/prepare_class_data.py --input "path/to/input_dataset"
```

This extracts class-0 boxes into `detect_two_stage/crops/`. Sort the crops into
`target/` and `non_target/` subfolders according to the intended binary task.
If an entire input belongs to one binary class, `--class-name target` or
`--class-name non_target` saves directly into that class folder.

```powershell
python detect_two_stage/generate_dataset.py --input "detect_two_stage/crops" --augment 2
python detect_two_stage/classifier.py
python detect_two_stage/export_onnx_classifier.py
python detect_two_stage/detect_two_stage.py --det-weights "best.pt" --source "path/to/images"
```

Dataset generation creates `detect_two_stage/dataset/{train,val,test}`, each with
`target/` and `non_target/` classes. It defaults to an 80/10/10 split with seed 42,
keeps crops from the same source filename together, and augments only training
images. At least three independent source images per class are required.

Classifier training reads `resnet18_imagenet.pth` from the repository root and saves
`detect_two_stage/yeni.pt`. Export writes `detect_two_stage/classifier.onnx`.
Inference defaults to that `.pt` checkpoint; `--cls-weights` selects another one.
Detection images and Grad-CAM outputs default to `runs/detect_two_stage/exp`.
The classifier currently selects checkpoints by training accuracy; validation and
test splits are loaded but not evaluated by its training loop.

Grad-CAM requires the `grad-cam` package, and classifier export requires ONNX
dependencies in addition to the repository environment. See
[detect_two_stage/README.md](detect_two_stage/README.md) for dataset layout,
single-image augmentation, and output options.

## Original YOLOv7 documentation

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/yolov7) using Gradio. Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)

## Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7
```

</details>

## Testing

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69730
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.55521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55937
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.63765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.53766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.73549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83868
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

Multiple GPU training

``` shell
# train p5 models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

Single GPU finetuning for custom dataset

``` shell
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

# finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```

## Re-parameterization

See [reparameterization.ipynb](tools/reparameterization.ipynb)

## Inference

On video:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>


## Export

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

</details>

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113

## Pose estimation

[`code`](https://github.com/WongKinYiu/yolov7/tree/pose) [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

See [keypoint.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/keypoint.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/pose.png" width="39%"/>
    </a>
</div>


## Instance segmentation (with NTU)

[`code`](https://github.com/WongKinYiu/yolov7/tree/mask) [`yolov7-mask.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

See [instance.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/mask.png" width="59%"/>
    </a>
</div>

## Instance segmentation

[`code`](https://github.com/WongKinYiu/yolov7/tree/u7/seg) [`yolov7-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt)

YOLOv7 for instance segmentation (YOLOR + YOLOv5 + YOLACT)

| Model | Test Size | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv7-seg** | 640 | **51.4%** | **69.4%** | **55.8%** | **41.5%** | **65.5%** | **43.7%** |

## Anchor free detection head

[`code`](https://github.com/WongKinYiu/yolov7/tree/u6) [`yolov7-u6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-u6.pt)

YOLOv7 with decoupled TAL head (YOLOR + YOLOv5 + YOLOv6)

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: |
| **YOLOv7-u6** | 640 | **52.6%** | **69.7%** | **57.3%** |


## Citation

```
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

```
@article{wang2023designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={Journal of Information Science and Engineering},
  year={2023}
}
```


## Teaser

YOLOv7-semantic & YOLOv7-panoptic & YOLOv7-caption

<div align="center">
    <a href="./">
        <img src="./figure/tennis.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_semantic.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_panoptic.png" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_caption.png" width="24%"/>
    </a>
</div>

YOLOv7-semantic & YOLOv7-detection & YOLOv7-depth (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_city.jpg" width="80%"/>
    </a>
</div>

YOLOv7-3d-detection & YOLOv7-lidar & YOLOv7-road (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_3d.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_lidar.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_road.jpg" width="30%"/>
    </a>
</div>


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
