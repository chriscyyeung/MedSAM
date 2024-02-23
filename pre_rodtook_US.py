"""RODTOOK dataset preprocessing script."""

import os
import glob
import argparse
import random
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

SEED = 2024
LOWER = np.array([15, 35, 35])
UPPER = np.array([70, 255, 255])
EPSILON = 0.0001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input-dir", 
        type=str,
        help="Path to directory containing UDIAT dataset."
    )
    parser.add_argument(
        "--train-output-dir", 
        type=str,
        help="Path to base directory for preprocessed training files."
    )
    parser.add_argument(
        "--test-output-dir", 
        type=str,
        help="Path to base directory for preprocessed testing files."
    )
    parser.add_argument(
        "--img-size", 
        type=int,
        default=224, 
        help="Resize images to square of this size. Default: 224."
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save a copy of cropped and extracted images/masks as png."
    )
    parser.add_argument(
        "--sanity-check", 
        action="store_true", 
        help="Save plots of image and ground truth for sanity check."
    )
    return parser.parse_args()


def main(args):
    train_imgs_path = os.path.join(args.train_output_dir, "imgs")
    train_gts_path = os.path.join(args.train_output_dir, "gts")
    test_imgs_path = os.path.join(args.test_output_dir, "imgs")
    test_gts_path = os.path.join(args.test_output_dir, "gts")
    os.makedirs(train_imgs_path, exist_ok=True)
    os.makedirs(train_gts_path, exist_ok=True)
    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_gts_path, exist_ok=True)

    if args.save_png:
        img_png_dir = os.path.join(args.input_dir, "imgs_png")
        gt_overlay_png_dir = os.path.join(args.input_dir, "gt_overlays_png")
        os.makedirs(os.path.join(args.input_dir, img_png_dir), exist_ok=True)
        os.makedirs(os.path.join(args.input_dir, gt_overlay_png_dir), exist_ok=True)

    img_dir = os.path.join(args.input_dir, "imgs")
    gt_overlay_dir = os.path.join(args.input_dir, "gt_overlays")

    # split data into train and test
    imgs = os.listdir(img_dir)
    n_train, n_test = round(len(imgs) * 0.8), round(len(imgs) * 0.2)
    splits = [True] * n_train + [False] * n_test
    random.Random(SEED).shuffle(splits)
    assert len(imgs) == len(splits), "Number of images and splits must match."

    for img_fn, split in tqdm(zip(imgs, splits), total=len(imgs)):
        img_id = f"{img_fn.split('-')[0]}-{img_fn.split('-')[1]}"  # Case-xxx
        gt_fn = glob.glob(os.path.join(gt_overlay_dir, img_id + "*"))[0]
        img = cv2.imread(os.path.join(img_dir, img_fn), cv2.IMREAD_GRAYSCALE)  # (H, W)
        gt_overlay = cv2.imread(gt_fn)  # (H, W, 3)

        # threshold contour by color
        gt_hsv = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2HSV)
        gt_mask = cv2.inRange(gt_hsv, LOWER, UPPER)

        # find contours from mask
        gt_seg = np.zeros_like(gt_overlay)
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_seg, contours, -1, (255, 255, 255), cv2.FILLED)
        gt_seg = cv2.cvtColor(gt_seg, cv2.COLOR_BGR2GRAY)  # (H, W)

        # find and draw contours one more time to extract largest contour
        seg_contours, _ = cv2.findContours(gt_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(seg_contours, key=cv2.contourArea)

        # approximate contour to smooth
        arclen = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, arclen * EPSILON, True)
        gt = np.zeros_like(gt_overlay)
        cv2.drawContours(gt, [approx], 0, (255, 255, 255), cv2.FILLED)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)  # (H, W)

        gt = cv2.resize(gt, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
        if args.save_png:
            cv2.imwrite(os.path.join(gt_overlay_png_dir, os.path.basename(gt_fn))[:-4] + ".png", gt)
            cv2.imwrite(os.path.join(img_png_dir, img_fn[:-4] + ".png"), img)

        # normalize images to [0, 1]
        if np.max(gt) > 1:
            gt = gt // 255
        assert np.max(gt) == 1 and np.min(gt) == 0, "Ground truth must be 0, 1."
        assert np.array_equal(gt, gt.astype(bool)), "Ground truth must be binary."

        if 1 < np.max(img) <= 255:
            img = img / 255
        assert np.max(img) <= 1.0 and np.min(img) >= 0.0, "Image must be in [0, 1]."

        if split:
            new_img_path = os.path.join(train_imgs_path, img_fn[:-4] + ".npy")
            new_gt_path = os.path.join(train_gts_path, os.path.basename(gt_fn)[:-4] + ".npy")
        else:
            new_img_path = os.path.join(test_imgs_path, img_fn[:-4] + ".npy")
            new_gt_path = os.path.join(test_gts_path, os.path.basename(gt_fn)[:-4] + ".npy")

        np.save(new_img_path, img[..., np.newaxis])  # (H, W, 1)
        np.save(new_gt_path, gt[..., np.newaxis])  # (H, W, 1)

    # plot and save image and gt
    if args.sanity_check:
        train_img_fn = os.listdir(train_imgs_path)[0]
        train_gt_fn = os.listdir(train_gts_path)[0]
        train_img = np.load(os.path.join(train_imgs_path, train_img_fn))
        train_gt = np.load(os.path.join(train_gts_path, train_gt_fn))
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(train_img, cmap="gray")
        axs[0].set_title("Image")
        axs[1].imshow(train_gt, cmap="gray")
        axs[1].set_title("Ground Truth")
        plt.savefig(os.path.join(args.train_output_dir, "sanity_check.png"))
        plt.close()

        test_img_fn = os.listdir(test_imgs_path)[0]
        test_gt_fn = os.listdir(test_gts_path)[0]
        test_img = np.load(os.path.join(test_imgs_path, test_img_fn))
        test_gt = np.load(os.path.join(test_gts_path, test_gt_fn))
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(test_img, cmap="gray")
        axs[0].set_title("Image")
        axs[1].imshow(test_gt, cmap="gray")
        axs[1].set_title("Ground Truth")
        plt.savefig(os.path.join(args.test_output_dir, "sanity_check.png"))
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
