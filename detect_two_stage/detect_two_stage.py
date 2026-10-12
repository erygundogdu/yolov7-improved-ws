import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from numpy import random
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


MEAN = [0.485, 0.456, 0.406]
STD= [0.229, 0.224, 0.225]
# -----------------------------
# Classifier
# -----------------------------
# class P4ResNetClassifier(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         backbone = models.resnet18(weights=None)
#         self.stem = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             backbone.relu,
#             backbone.maxpool,
#         )
#         self.layer1 = backbone.layer1
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#         self.layer4 = backbone.layer4
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         embed = self.avgpool(x)
#         embed = torch.flatten(embed, 1)

#         logits = self.fc(embed)
#         return logits, embed
class P4ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # P4/C4 feature level
        self.layer4 = backbone.layer4
        
        self.multi = True
        proj_dim = 256
        self.fc5 = nn.Linear(512, proj_dim)
        self.fc4 = nn.Linear(256, proj_dim)
        self.fc3 = nn.Linear(128, proj_dim)
        #self.emb_fc = nn.Linear(512, EMBED_DIM)
        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.cls = nn.Linear(proj_dim, num_classes)
        #self.fc = nn.Linear(512,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        out = torch.flatten(self.avg_pool(f5), 1)
        logits5 = self.fc5(out)
        
        f4 = torch.flatten(self.avg_pool(f4), 1)
        logits4 = self.fc4(f4)
        f3 = torch.flatten(self.avg_pool(f3), 1)
        logits3 = self.fc3(f3)
        #cat = F.normalize(torch.cat([logits3, logits4, logits5], dim=1),dim=1,p=2)
        cat = torch.cat([logits3, logits4, logits5], dim=1)
        w = self.scale_attn(cat) ##b,3
        fused = (w[:, 0:1] * logits3 +w[:, 1:2] * logits4 +w[:, 2:3] * logits5)

        logits = self.cls(fused)
        return logits, logits5
        # else:
        #     logits = self.fc(logits5)
        #     return logits,logits 

class P4ResNetClassifier1(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # P4/C4 feature level
        self.layer4 = backbone.layer4
        
        self.multi = True
        proj_dim = 256
        self.fc5 = nn.Linear(512, proj_dim)
        self.fc4 = nn.Linear(256, proj_dim)
        self.fc3 = nn.Linear(128, proj_dim)
        #self.emb_fc = nn.Linear(512, EMBED_DIM)
        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.cls = nn.Linear(3*proj_dim, num_classes)
        #self.fc = nn.Linear(512,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        out = torch.flatten(self.avg_pool(f5), 1)
        logits5 = self.fc5(out)
        
        f4 = torch.flatten(self.avg_pool(f4), 1)
        logits4 = self.fc4(f4)
        f3 = torch.flatten(self.avg_pool(f3), 1)
        logits3 = self.fc3(f3)
        #cat = F.normalize(torch.cat([logits3, logits4, logits5], dim=1),dim=1,p=2)
        logits3 = F.normalize(logits3,p=2, dim=1)
        logits4 = F.normalize(logits4,p=2, dim=1)
        logits5 = F.normalize(logits5,p=2, dim=1)
        cat = torch.cat([logits3, logits4, logits5], dim=1)
        
        logits = self.cls(cat)
        return logits, logits5
        # else:
        #     logits = self.fc(logits5)
        #     return logits,logits 

CLS_IMG_SIZE = 160
ALPHA = 2.0

cls_tfms = transforms.Compose([
    transforms.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN,
                         std=STD),
])


class ClassifierGradCAMWrapper(nn.Module):
    def __init__(self, modelc):
        super().__init__()
        self.modelc = modelc

    def forward(self, x):
        logits, _ = self.modelc(x)
        return logits


def expand_box_xyxy(x1, y1, x2, y2, img_w, img_h, alpha=2.0):
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = alpha * max(bw, bh)

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0
    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))


def save_classifier_gradcam(crop_bgr, modelc, device, save_path, target_layer_name="layer3"):
    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)

    x = cls_tfms(pil_crop).unsqueeze(0).to(device)

    rgb_vis = np.array(
        pil_crop.resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
    ).astype(np.float32) / 255.0

    wrapped_model = ClassifierGradCAMWrapper(modelc).to(device)
    wrapped_model.eval()

    if target_layer_name == "layer2":
        target_layers = [wrapped_model.modelc.layer2[-1]]
    elif target_layer_name == "layer3":
        target_layers = [wrapped_model.modelc.layer3[-1]]
    elif target_layer_name == "layer4":
        target_layers = [wrapped_model.modelc.layer4[-1]]
    else:
        raise ValueError("target_layer_name must be 'layer2', 'layer3', or 'layer4'")

    targets = [BinaryClassifierOutputTarget(1)]

    with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=x, targets=targets)[0]

    cam_image = show_cam_on_image(rgb_vis, grayscale_cam, use_rgb=True)
    cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), cam_bgr)

    with torch.no_grad():
        logits, _ = modelc(x)
        prob = torch.sigmoid(logits).item()

    return prob


def classify_crop(im0, xyxy, modelc, device, alpha=2.0, save_cam_path=None, cam_layer="layer3"):
    h, w = im0.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx1, cy1, cx2, cy2 = expand_box_xyxy(x1, y1, x2, y2, w, h, alpha=alpha)

    crop = im0[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return "no", 0.0

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    x = cls_tfms(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit, _ = modelc(x)
        prob = torch.sigmoid(logit).item()
        pred_idx = 1 if prob >= 0.5 else 0
        conf = prob

    if save_cam_path is not None:
        save_classifier_gradcam(
            crop_bgr=crop,
            modelc=modelc,
            device=device,
            save_path=save_cam_path,
            target_layer_name=cam_layer,
        )

    label = "yes" if pred_idx == 1 else "no"

    return label, conf


def draw_yes_no(im0, xyxy, yn_label, yn_conf):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    text = f"{yn_label.upper()} {yn_conf:.2f}"
    color = (0, 255, 0) if yn_label == "yes" else (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    tx2 = x2
    tx1 = max(0, tx2 - tw - 6)
    ty1 = max(0, y1)
    ty2 = min(im0.shape[0] - 1, ty1 + th + baseline + 6)

    cv2.rectangle(im0, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        im0,
        text,
        (tx1 + 3, ty2 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def detect():
    source = opt.source
    save_img = not opt.nosave

    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    cam_dir = save_dir / "classifier_gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"

    model = attempt_load(opt.det_weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)

    if half:
        model.half()

    model.eval()

    modelc = P4ResNetClassifier(num_classes=1)
    modelc.load_state_dict(torch.load(opt.cls_weights, map_location=device))
    modelc = modelc.to(device).eval()

    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    vid_path, vid_writer = None, None

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        view_img = opt.view_img
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != "cpu":
        dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
        dummy = dummy.half() if half else dummy.float()
        _ = model(dummy)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()

        with torch.no_grad():
            pred = model(img)[0]

        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms
        )

        t2 = time_synchronized()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)

            s += "%gx%g " % img.shape[2:]
            box_results = []

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls.item())
                    conf = float(conf.item())

                    label = f"{names[cls]} {conf:.2f}"

                    cam_save_path = cam_dir / f"{p.stem}_frame{frame}_box{len(box_results)}_{opt.cam_layer}.jpg"

                    yn_label, yn_conf = classify_crop(
                        im0,
                        xyxy,
                        modelc,
                        device,
                        alpha=opt.alpha,
                        save_cam_path=cam_save_path,
                        cam_layer=opt.cam_layer
                    )

                    box_results.append({
                        "xyxy": xyxy,
                        "det_label": label,
                        "det_cls": cls,
                        "yn_label": yn_label,
                        "yn_conf": yn_conf,
                    })

            print(f"{s}Done. ({t2 - t1:.3f}s)")

            for b in box_results:
                xyxy = b["xyxy"]
                det_label = b["det_label"]
                det_cls = b["det_cls"]
                yn_label = b["yn_label"]
                yn_conf = b["yn_conf"]

                plot_one_box(
                    xyxy,
                    im0,
                    label=det_label,
                    color=colors[det_cls],
                    line_thickness=opt.line_thickness
                )

                draw_yes_no(im0, xyxy, yn_label, yn_conf)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path

                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"

                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h)
                        )

                    vid_writer.write(im0)

    print(f"Results saved to {save_dir}")
    print(f"Classifier Grad-CAMs saved to {cam_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--det-weights", type=str, required=True, help="detector weights .pt")
    parser.add_argument("--cls-weights", type=str, default=str(Path(__file__).resolve().parent / "yeni.pt"), help="classifier weights .pt (default: detect_two_stage/yeni.pt)")
    parser.add_argument("--source", type=str, required=True, help="image folder / video / stream")
    parser.add_argument("--img-size", type=int, default=1664, help="detector inference size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="detector confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="detector NMS IoU threshold")
    parser.add_argument("--alpha", type=float, default=2.0, help="bbox expansion factor for verifier crop")
    parser.add_argument("--device", default="", help="cuda device")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--nosave", action="store_true", help="do not save results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter detection classes")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--project", default="runs/detect_two_stage", help="save dir")
    parser.add_argument("--name", default="exp", help="save subdir")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite existing dir")
    parser.add_argument("--line-thickness", type=int, default=2, help="bbox thickness")
    parser.add_argument("--cam-layer", type=str, default="layer3", choices=["layer2", "layer3", "layer4"], help="classifier layer used for Grad-CAM")

    opt = parser.parse_args()

    detect()
