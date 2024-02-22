"""OASBUD dataset preprocessing script."""

import os
import argparse
import random
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.signal import hilbert

SEED = 2024
DB_THRESHOLD = -50.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", 
        "--input-mat", 
        type=str,
        help="Path to mat file containing dataset."
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

    mat = io.loadmat(args.input_mat, squeeze_me=True)
    data = mat["data"]

    # split data into train and test
    n_malignant = len(data["id"] == 1)
    n_train, n_test = round(n_malignant * 0.8), round(n_malignant * 0.2)
    splits = [True] * n_train + [False] * n_test
    random.Random(SEED).shuffle(splits)
    assert n_malignant == len(splits), "Number of images and splits must match."

    for i in tqdm(range(len(data["id"]))):
        if data["class"][i] == 1:
            # from https://github.com/tensorflow/datasets/pull/2428/files
            # hilbert transform, log compression, dB thresholding
            raw_rf = data["rf1"][i]  # (H, W)
            envelope_im = np.abs(hilbert(raw_rf))
            compress_im = 20 * np.log10(envelope_im / np.max(envelope_im))
            compress_im[compress_im < DB_THRESHOLD] = DB_THRESHOLD

            img = cv2.resize(compress_im, (args.img_size, args.img_size), interpolation=cv2.INTER_CUBIC)
            img = (img - np.min(img)) / np.ptp(img)  # normalize to [0, 1]
            assert np.max(img) <= 1.0 and np.min(img) >= 0.0, "Image must be in [0, 1]."

            # read ground truth
            gt = data["roi1"][i]  # (H, W)
            gt = cv2.resize(gt, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
            assert np.max(gt) == 1 and np.min(gt) == 0, "Ground truth must be 0, 1."

            img_id = data["id"][i]
            if splits[i]:
                new_img_path = os.path.join(train_imgs_path, img_id + ".npy")
                new_gt_path = os.path.join(train_gts_path, img_id + ".npy")
            else:
                new_img_path = os.path.join(test_imgs_path, img_id + ".npy")
                new_gt_path = os.path.join(test_gts_path, img_id + ".npy")

            np.save(new_img_path, img[..., np.newaxis])  # (H, W, 1)
            np.save(new_gt_path, gt[..., np.newaxis])  # (H, W, 1)

    # plot and save image and gt
    if args.sanity_check:
        train_img_fn = os.listdir(train_imgs_path)[0]
        train_img = np.load(os.path.join(train_imgs_path, train_img_fn))
        train_gt = np.load(os.path.join(train_gts_path, train_img_fn))
        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(train_img, cmap="gray")
        axs[0].set_title("Image")
        axs[1].imshow(train_gt, cmap="gray")
        axs[1].set_title("Ground Truth")
        plt.savefig(os.path.join(args.train_output_dir, "sanity_check.png"))
        plt.close()

        test_img_fn = os.listdir(test_imgs_path)[0]
        test_img = np.load(os.path.join(test_imgs_path, test_img_fn))
        test_gt = np.load(os.path.join(test_gts_path, test_img_fn))
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
