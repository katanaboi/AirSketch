"""Bulk-import MediaPipe hand landmarks from image folders into the dataset CSV.

Reads images from a folder (ImageFolder convention, or flat with --label),
runs MediaPipe hand detection on each, and appends 42-feature rows to the
same CSV that live capture writes to.

Usage:
    # ImageFolder-style: each subfolder is a class label
    python import_images.py --folder path/to/asl_alphabet_train

    # Flat folder, explicit label
    python import_images.py --folder path/to/letter_a_only --label a

    # Inspect without writing
    python import_images.py --folder path/to/x --dry-run
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from utils import extract_hand_landmark_points


DEFAULT_DATASET = "data/hand_landmarks_dataset.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PROGRESS_EVERY = 100


def csv_header():
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")
    return header


def find_images(folder):
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def find_class_folders(root):
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir())


def existing_label_counts(dataset_path):
    if not os.path.exists(dataset_path):
        return Counter()
    try:
        labels = pd.read_csv(dataset_path, usecols=["label"])["label"]
        return Counter(labels.astype(str).tolist())
    except (ValueError, pd.errors.EmptyDataError):
        return Counter()


def resolve_targets(folder, explicit_label):
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        sys.exit(f"Folder not found or not a directory: {folder}")

    if explicit_label:
        return [(explicit_label, folder)]

    subs = find_class_folders(folder)
    if not subs:
        sys.exit(
            f"No subfolders found under {folder}. "
            "Pass --label <name> to treat it as a flat single-class folder."
        )
    return [(p.name, p) for p in subs]


def open_writer(dataset_path, dry_run):
    if dry_run:
        return None, None
    parent = os.path.dirname(dataset_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.exists(dataset_path)
    fh = open(dataset_path, "a", newline="")
    writer = csv.writer(fh)
    if not file_exists:
        writer.writerow(csv_header())
    return fh, writer


def process_folder(hands, folder, label, writer, flip, dry_run):
    images = find_images(folder)
    added = 0
    no_hand = 0
    unreadable = 0

    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            unreadable += 1
            continue
        if flip:
            img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            no_hand += 1
        else:
            for hand_landmarks in results.multi_hand_landmarks:
                points = extract_hand_landmark_points(hand_landmarks)
                if not dry_run:
                    writer.writerow(points + [label])
                added += 1

        if idx % PROGRESS_EVERY == 0:
            print(
                f"  [{label}] {idx}/{len(images)} processed "
                f"(added {added}, no-hand {no_hand}, unreadable {unreadable})"
            )

    return {
        "scanned": len(images),
        "added": added,
        "no_hand": no_hand,
        "unreadable": unreadable,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--folder", required=True,
        help="Dataset root (subfolders = class labels) or flat folder with --label.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Single label for flat-folder mode. If set, --folder is treated as one class.",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Target CSV path (default {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--flip", action="store_true",
        help="Horizontally flip each image before MediaPipe (use when the source is mirrored).",
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="MediaPipe min_detection_confidence (default 0.5).",
    )
    parser.add_argument(
        "--max-hands", type=int, choices=[1, 2], default=1,
        help="Number of hands to save per image (default 1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Walk folders and run detection, but write nothing.",
    )
    args = parser.parse_args()

    targets = resolve_targets(args.folder, args.label)
    total_images = sum(len(find_images(f)) for _, f in targets)

    existing = existing_label_counts(args.dataset)
    new_labels = [lbl for lbl, _ in targets if lbl not in existing]

    print(f"Dataset target: {args.dataset}{'  [DRY RUN]' if args.dry_run else ''}")
    print(
        f"static_image_mode=True | max_hands={args.max_hands} | "
        f"min_detection_confidence={args.min_detection_confidence} | flip={args.flip}"
    )
    print(f"Planned: {len(targets)} label(s), {total_images} image(s) total\n")

    if new_labels:
        print("New labels not yet present in the CSV:")
        for lbl in new_labels:
            print(f"  - {lbl}")
        print("  (Double-check spelling/case before a non-dry-run import.)\n")

    mp_hands = mp.solutions.hands
    fh, writer = open_writer(args.dataset, args.dry_run)

    totals = {}
    try:
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detection_confidence,
        ) as hands:
            for label, folder in targets:
                print(f"[{label}] folder={folder}")
                stats = process_folder(
                    hands, folder, label, writer, args.flip, args.dry_run,
                )
                totals[label] = stats
                print(
                    f"  -> scanned {stats['scanned']} | added {stats['added']} | "
                    f"no-hand {stats['no_hand']} | unreadable {stats['unreadable']}\n"
                )
    finally:
        if fh is not None:
            fh.close()

    print("Summary:")
    verb = "would add" if args.dry_run else "added"
    for label, stats in totals.items():
        pre = existing.get(label, 0)
        after = pre + (stats["added"] if not args.dry_run else 0)
        print(
            f"  {label}: {verb} {stats['added']} rows "
            f"(prev {pre} -> now {after}), "
            f"scanned {stats['scanned']} "
            f"(no-hand {stats['no_hand']}, unreadable {stats['unreadable']})"
        )

    if args.dry_run:
        print("\nDry run complete. No rows written.")


if __name__ == "__main__":
    main()
