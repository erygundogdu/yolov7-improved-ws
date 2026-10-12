import argparse
import time
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, letterbox
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, load_classifier, select_device, time_synchronized


def _make_slice_starts(length, slice_length, overlap_ratio):
    if slice_length >= length:
        return [0]

    overlap_ratio = max(0.0, min(0.99, overlap_ratio))
    step = max(int(slice_length * (1.0 - overlap_ratio)), 1)
    starts = list(range(0, max(length - slice_length, 0) + 1, step))
    if not starts:
        starts = [0]
    last = starts[-1]
    if last + slice_length < length:
        starts.append(length - slice_length)
    return sorted(set(starts))


def _generate_slices(h, w, slice_h, slice_w, overlap_h, overlap_w):
    ys = _make_slice_starts(h, slice_h, overlap_h)
    xs = _make_slice_starts(w, slice_w, overlap_w)
    return [(x, y, min(x + slice_w, w), min(y + slice_h, h)) for y in ys for x in xs]


def _run_sliced_inference(model, im0, device, half, opt, stride):
    h, w = im0.shape[:2]
    slice_h = max(1, min(opt.slice_size, h))
    slice_w = max(1, min(opt.slice_size, w))

    tiles = _generate_slices(
        h=h,
        w=w,
        slice_h=slice_h,
        slice_w=slice_w,
        overlap_h=opt.slice_overlap_h,
        overlap_w=opt.slice_overlap_w,
    )

    merged_dets = []
    inf_time = 0.0
    nms_time = 0.0

    for x1, y1, x2, y2 in tiles:
        crop = im0[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        img = letterbox(crop, opt.img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        img = img.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        det = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )[0]
        t3 = time_synchronized()

        inf_time += t2 - t1
        nms_time += t3 - t2

        if len(det):
            det = det.clone()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], crop.shape).round()
            det[:, [0, 2]] += x1
            det[:, [1, 3]] += y1
            merged_dets.append(det)

    if len(merged_dets):
        det = torch.cat(merged_dets, 0)
        t4 = time_synchronized()
        c = det[:, 5:6] * (0 if opt.agnostic_nms else 4096)
        keep = torchvision.ops.nms(det[:, :4] + c, det[:, 4], opt.slice_merge_iou_thres)
        det = det[keep]
        t5 = time_synchronized()
        nms_time += t5 - t4
    else:
        det = torch.zeros((0, 6), device=device)

    return det, tiles, inf_time, nms_time


def _draw_grid_and_boxes(image_bgr, det, tiles):
    out = image_bgr.copy()

    for x1, y1, x2, y2 in tiles:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for row in det.cpu().numpy():
        x1, y1, x2, y2, conf, cls = row
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)} {conf:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
    )
    save_img = not opt.nosave and not source.endswith(".txt")
    webcam = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    print("**********************************************************", imgsz)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        batch_items = []
        if webcam:
            for i in range(len(im0s)):
                batch_items.append((i, path[i], im0s[i].copy(), dataset.count))
        else:
            batch_items.append((0, path, im0s, getattr(dataset, "frame", 0)))

        for i, p_raw, im0, frame in batch_items:
            t1 = time_synchronized()
            det, tiles, infer_dt, nms_dt = _run_sliced_inference(model, im0, device, half, opt, stride)
            t2 = time_synchronized()

            if classify:
                img_for_cls = torch.zeros(1, 3, imgsz, imgsz, device=device)
                det = apply_classifier([det], modelc, img_for_cls, im0)[0]

            if webcam:
                p, s = Path(p_raw), f"{i}: "
            else:
                p, s = Path(p_raw), ""

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if opt.draw_windows and (save_img or view_img):
                im0 = _draw_grid_and_boxes(im0, det, tiles)

            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if (save_img or view_img) and not opt.draw_windows:
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            print(
                f"{s}Done. ({(1E3 * infer_dt):.1f}ms) Inference, "
                f"({(1E3 * nms_dt):.1f}ms) NMS, ({(1E3 * (t2 - t1)):.1f}ms) SAHI total"
            )

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
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
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        _ = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)")
    parser.add_argument("--source", type=str, default="inference/images", help="source")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--draw-windows", action="store_true", help="draw red slice windows and green detection boxes")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_false", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent / "runs"), help="save results to project/name (default: sahi/runs)")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--no-trace", action="store_true", help="do`t trace model")

    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size in pixels")
    parser.add_argument("--slice-overlap-h", type=float, default=0.2, help="SAHI overlap ratio for height (0-1)")
    parser.add_argument("--slice-overlap-w", type=float, default=0.2, help="SAHI overlap ratio for width (0-1)")
    parser.add_argument(
        "--slice-merge-iou-thres",
        type=float,
        default=0.30,
        help="IOU threshold for merged full-image NMS over slice detections",
    )

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ["yolov7.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
