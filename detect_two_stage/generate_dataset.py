import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise ValueError(f"No images found in: {directory}")
    return images


def source_key(path):
    return re.split(r"_cls0_obj\d+_a", path.stem, maxsplit=1)[0]


def augment(image, rng, img_size):
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.2:
        image = ImageOps.flip(image)
    return image


def check_output(output, inputs):
    output = Path(output).expanduser().resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output must be separate from the source directory: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output already exists and is not empty: {output}. Choose a new --out directory.")
    return output


def generate_target_dataset(image_path, n, out_root, img_size=160, seed=42):
    image_path = Path(image_path).expanduser().resolve()
    if n < 1 or img_size < 1:
        raise ValueError("--n and --img-size must be positive.")
    output = check_output(out_root, [image_path])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    target_dir = output / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        augment(image, rng, img_size).save(target_dir / f"target_{i:06d}.jpg", quality=95)
    print(f"Generated {n} target images under: {target_dir}")


def create_dataset(target, non_target, output, ratios, seed=42, augmentations=0, img_size=160):
    if len(ratios) != 3 or any(not 0 < r < 1 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--split must contain three positive fractions adding up to 1.")
    if augmentations < 0 or img_size < 1:
        raise ValueError("--augment must be nonnegative and --img-size must be positive.")
    sources = {"target": list_images(target), "non_target": list_images(non_target)}
    if set(sources["target"]) & set(sources["non_target"]):
        raise ValueError("Target and non-target inputs must contain different images.")
    output = check_output(output, [target, non_target])
    groups = {}
    for label, paths in sources.items():
        groups[label] = {}
        for path in paths:
            groups[label].setdefault(source_key(path), []).append(path)
        if len(groups[label]) < 3:
            raise ValueError(f"{label} needs at least 3 independent source images for train/val/test.")

    assignment = None
    for attempt in range(1000):
        candidate = {}
        rng = random.Random(seed + attempt)
        for label in ("target", "non_target"):
            keys = sorted(groups[label])
            rng.shuffle(keys)
            train_count = max(1, min(len(keys) - 2, int(len(keys) * ratios[0])))
            val_count = max(1, min(len(keys) - train_count - 1, int(len(keys) * ratios[1])))
            for i, key in enumerate(keys):
                split = "train" if i < train_count else "val" if i < train_count + val_count else "test"
                candidate.setdefault(key, split)
        if all(any(candidate[key] == split for key in groups[label])
               for label in groups for split in ("train", "val", "test")):
            assignment = candidate
            break
    if assignment is None:
        raise ValueError("Cannot form nonempty splits for both classes with these source groups.")

    for paths in sources.values():
        for path in paths:
            with Image.open(path) as image:
                image.verify()

    rng = random.Random(seed)
    counts = {split: {label: 0 for label in sources} for split in ("train", "val", "test")}
    for split in counts:
        for label in sources:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    for label, paths in sources.items():
        for index, path in enumerate(paths):
            split = assignment[source_key(path)]
            destination = output / split / label
            shutil.copy2(path, destination / f"{index:06d}_{path.name}")
            counts[split][label] += 1
            if split == "train" and augmentations:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                for i in range(augmentations):
                    augment(image, rng, img_size).save(destination / f"aug_{index:06d}_{i:04d}.jpg", quality=95)
                    counts[split][label] += 1
    for split, labels in counts.items():
        print(f"{split}: " + ", ".join(f"{label}={count}" for label, count in labels.items()))
    print(f"Dataset: {output}")


def main():
    parser = argparse.ArgumentParser(description="Build classifier train/val/test splits or augment a single target image.")
    parser.add_argument("--input", type=Path, help="Directory containing target/ and non_target/")
    parser.add_argument("--target", type=Path, help="Directory of target images")
    parser.add_argument("--non-target", type=Path, help="Directory of non-target images")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to detect_two_stage/dataset")
    parser.add_argument("--split", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=int, default=0, help="Extra augmented images per training image")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--image", type=Path, help="Single-image augmentation mode; does not create splits")
    parser.add_argument("--n", type=int, help="Number of images for single-image augmentation mode")
    args = parser.parse_args()
    try:
        if args.image:
            if args.input or args.target or args.non_target or args.augment:
                parser.error("--image cannot be combined with dataset input options or --augment.")
            if args.n is None:
                parser.error("--image requires --n.")
            generate_target_dataset(args.image, args.n, args.out or BASE_DIR / "generated_dataset", args.img_size, args.seed)
        else:
            if args.n is not None:
                parser.error("--n requires --image. Use --augment for training augmentation.")
            if args.input:
                if args.target or args.non_target:
                    parser.error("Use --input or --target with --non-target, not both.")
                args.target = args.input.expanduser() / "target"
                args.non_target = args.input.expanduser() / "non_target"
            if not args.target or not args.non_target:
                parser.error("Supply --input PATH or both --target PATH and --non-target PATH.")
            create_dataset(args.target, args.non_target, args.out or BASE_DIR / "dataset", args.split,
                           args.seed, args.augment, args.img_size)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
