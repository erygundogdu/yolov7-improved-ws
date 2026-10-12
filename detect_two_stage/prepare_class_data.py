import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image

ALPHA = 1.25
OUTPUT_SIZE = 160
KEEP_CLASS_ID = 0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x_c = x_center * img_w
    y_c = y_center * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = x_c - bw / 2.0
    y1 = y_c - bh / 2.0
    x2 = x_c + bw / 2.0
    y2 = y_c + bh / 2.0
    return x1, y1, x2, y2


def expand_to_square(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    alpha: float,
) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = alpha * max(bw, bh)

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0
    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(img_w), nx2)
    ny2 = min(float(img_h), ny2)

    cur_w = nx2 - nx1
    cur_h = ny2 - ny1
    side2 = max(cur_w, cur_h)

    cx2 = (nx1 + nx2) / 2.0
    cy2 = (ny1 + ny2) / 2.0

    nx1 = max(0.0, cx2 - side2 / 2.0)
    ny1 = max(0.0, cy2 - side2 / 2.0)
    nx2 = min(float(img_w), cx2 + side2 / 2.0)
    ny2 = min(float(img_h), cy2 + side2 / 2.0)

    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not label_path.exists():
        return rows

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            rows.append((cls_id, xc, yc, w, h))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare classifier crops from YOLO labels.")
    parser.add_argument("--input", type=Path, help="Dataset directory containing images/ and labels/, or the images/ directory")
    parser.add_argument("--labels", type=Path, help="Override the labels directory")
    parser.add_argument("--output", type=Path, help="Override the output directory")
    parser.add_argument("--class-name", choices=("target", "non_target"), help="Class of all extracted crops; otherwise leave crops unlabelled")
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        try:
            entered_path = input("Dataset or images directory: ").strip().strip('\"').strip("'")
        except EOFError:
            parser.error("No input path supplied. Use --input PATH.")
        if not entered_path:
            parser.error("Input path cannot be empty.")
        input_path = Path(entered_path)

    input_path = input_path.expanduser()
    if (input_path / "images").is_dir():
        dataset_dir = input_path
        images_dir = dataset_dir / "images"
    else:
        images_dir = input_path
        dataset_dir = input_path.parent
    labels_dir = args.labels.expanduser() if args.labels else dataset_dir / "labels"
    output_dir = args.output.expanduser() if args.output else Path(__file__).resolve().parent / "crops"
    if args.class_name:
        output_dir = output_dir / args.class_name

    for name, directory in (("Images", images_dir), ("Labels", labels_dir)):
        if not directory.is_dir():
            parser.error(f"{name} directory does not exist: {directory}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f"Output already exists and is not empty: {output_dir}. Choose a new --output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    total_saved = 0
    total_images = 0
    total_class0_boxes = 0
    missing_labels = 0

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        labels = load_yolo_labels(label_path)
        class0_labels = [row for row in labels if row[0] == KEEP_CLASS_ID]

        if len(class0_labels) == 0:
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        total_images += 1
        total_class0_boxes += len(class0_labels)

        for obj_idx, (_, xc, yc, w, h) in enumerate(class0_labels):
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            cx1, cy1, cx2, cy2 = expand_to_square(x1, y1, x2, y2, img_w, img_h, ALPHA)

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = img.crop((cx1, cy1, cx2, cy2))
            crop = crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)

            save_name = (
                f"{img_path.stem}"
                f"_cls0_obj{obj_idx}"
                f"_a{ALPHA:.1f}"
                f"_x1{cx1}_y1{cy1}_x2{cx2}_y2{cy2}.jpg"
            )
            save_path = output_dir / save_name
            crop.save(save_path, quality=95)
            total_saved += 1

    print("Done.")
    print(f"Images with class-0 boxes : {total_images}")
    print(f"Total class-0 boxes       : {total_class0_boxes}")
    print(f"Saved crops               : {total_saved}")
    print(f"Missing label files       : {missing_labels}")
    print(f"Output folder             : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
