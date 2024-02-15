import os
import glob
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input-dir", 
        type=str,
        help="Path to directory containing images and masks (in png format)."
    )
    parser.add_argument(
        "-o", 
        "--output-dir", 
        type=str,
        help="Path to base directory for preprocessed files."
    )
    parser.add_argument(
        "--patient-ids", 
        nargs="*", 
        default="all",
        help="List of patient IDs to process. Default: use all."
    )
    parser.add_argument(
        "--crop-size", 
        type=int,
        help="Crop size of input images (square). Omit to keep original size."
    )
    parser.add_argument(
        "--img-size", 
        type=int,
        default=224, 
        help="Resize images to square of this size. Default: 224."
    )
    parser.add_argument(
        "--log-empty-gt", 
        action="store_true",
        help="Save IDs of images with empty ground truth to text file."
    )
    parser.add_argument(
        "--sanity-check", 
        action="store_true", 
        help="Save plots of image and ground truth for sanity check."
    )
    return parser.parse_args()


def crop(img, size):
    """Remove black borders and crop ultrasound image to square."""
    img_w = img.shape[1]
    start_w = (img_w - size) // 2
    return img[:size, start_w:start_w + size]


def main(args):
    imgs_path = os.path.join(args.output_dir, "imgs")
    gts_path = os.path.join(args.output_dir, "gts")
    os.makedirs(imgs_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    imgs = glob.glob(os.path.join(args.input_dir, "*ultrasound.png"))
    gts = glob.glob(os.path.join(args.input_dir, "*segmentation.png"))
    assert len(imgs) == len(gts), "Number of images and segmentations do not match."

    patient_ids = []
    empty_gt_ids = []

    for img_fn, gt_fn in tqdm(zip(imgs, gts), total=len(imgs)):
        patient_id = os.path.basename(img_fn).split("_")[0]
        if args.patient_ids != "all" and patient_id not in args.patient_ids:
            continue
        if patient_id not in patient_ids:
            patient_ids.append(patient_id)  # for sanity check later

        img_num = os.path.basename(img_fn).split("_")[1]
        img_id = f"{patient_id}_{img_num}"

        gt = cv2.imread(gt_fn, cv2.IMREAD_GRAYSCALE)
        if np.max(gt) == 0:
            if args.log_empty_gt:
                empty_gt_ids.append(img_id)
            continue  # skip empty ground truth
        gt = np.flip(gt, axis=0)
        if args.img_size:
            gt = crop(gt, args.crop_size)
        gt = cv2.resize(gt, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        if np.max(gt) > 1:
            gt = gt // 255  # convert to binary
        assert np.max(gt) == 1 and np.min(gt) == 0, "Ground truth must be 0, 1."

        img = cv2.imread(img_fn)  # (H, W, 3)
        img = np.flip(img, axis=0)
        if args.img_size:
            img = crop(img, args.crop_size)
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
        if 1 < np.max(img) <= 255:
            img = img / 255  # normalize to [0, 1]
        assert np.max(img) <= 1.0 and np.min(img) >= 0.0, "Image must be in [0, 1]."

        new_img_path = os.path.join(imgs_path, img_id + ".npy")
        new_gt_path = os.path.join(gts_path, img_id + ".npy")
        np.save(new_img_path, img)
        np.save(new_gt_path, gt)

    if args.log_empty_gt:
        with open(os.path.join(args.output_dir, "empty_gt_ids.txt"), "a") as f:
            f.write("\n".join(empty_gt_ids))

    # plot and save image and gt
    if args.sanity_check:
        for pid in patient_ids:
            img_id = [f for f in os.listdir(imgs_path) if f.startswith(pid)][0]  # get first image
            img = np.load(os.path.join(imgs_path, img_id))
            gt = np.load(os.path.join(gts_path, img_id))

            _, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img, cmap="gray")
            axs[0].set_title("Image")
            axs[1].imshow(gt, cmap="gray")
            axs[1].set_title("Ground Truth")
            plt.savefig(os.path.join(args.output_dir, f"{img_id[:-4]}_sanitycheck.png"))
            plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
