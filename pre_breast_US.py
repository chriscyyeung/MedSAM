"""BrEaST dataset preprocessing script."""

import os
import argparse
import random
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 2024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input-dir", 
        type=str,
        help="Path to directory containing BrEaST dataset."
    )
    parser.add_argument(
        "--data-excel",
        type=str,
        help="Path to excel file containing image clinical data."
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
        "--type", 
        type=str, 
        help="Tumor class to preprocess (benign or malignant)."
    )
    parser.add_argument(
        "--img-size", 
        type=int,
        default=224, 
        help="Resize images to square of this size. Default: 224."
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

    # read excel with tumor classification
    df = pd.read_excel(args.data_excel, dtype=str)
    img_fns = df[df["Classification"].str.lower() == args.type]["Image_filename"].tolist()
    gt_fns = df[df["Classification"].str.lower() == args.type]["Mask_tumor_filename"].tolist()
    assert len(img_fns) == len(gt_fns), "Number of images and ground truths must match."

    # split data into train and test
    n_train, n_test = round(len(img_fns) * 0.8), round(len(img_fns) * 0.2)
    splits = [True] * n_train + [False] * n_test
    random.Random(SEED).shuffle(splits)
    assert len(img_fns) == len(splits), "Number of images and splits must match."

    for img_fn, gt_fn, split in tqdm(zip(img_fns, gt_fns, splits), total=len(img_fns)):
        gt = cv2.imread(os.path.join(args.input_dir, gt_fn), cv2.IMREAD_GRAYSCALE)  # (H, W)
        gt = cv2.resize(gt, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        if np.max(gt) > 1:
            gt = gt // 255  # convert to binary
        assert np.max(gt) == 1 and np.min(gt) == 0, "Ground truth must be 0, 1."

        img = cv2.imread(os.path.join(args.input_dir, img_fn), cv2.IMREAD_GRAYSCALE)  # (H, W)
        img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
        if 1 < np.max(img) <= 255:
            img = img / 255  # normalize to [0, 1]
        assert np.max(img) <= 1.0 and np.min(img) >= 0.0, "Image must be in [0, 1]."

        if split:
            new_img_path = os.path.join(train_imgs_path, img_fn[:-4] + ".npy")
            new_gt_path = os.path.join(train_gts_path, gt_fn[:-4] + ".npy")
        else:
            new_img_path = os.path.join(test_imgs_path, img_fn[:-4] + ".npy")
            new_gt_path = os.path.join(test_gts_path, gt_fn[:-4] + ".npy")

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
