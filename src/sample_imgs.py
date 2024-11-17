import os
import cv2
import torch
from pathlib import Path
import albumentations as A
import torchvision.transforms as transforms
from torchvision.transforms import v2
from plot_utils import *

script_path = Path(__file__).resolve().parent


if __name__ == '__main__':
    dataset_path = f'{script_path}/../data'

    # Rand Augmentation
    for magnitude in [1, 5, 9, 13, 17, 21]:
        print(f"magnitude: {magnitude}")
        augmented_dir = f"{dataset_path}/RandAugment/augmented_tensor"
        sample_img_dir = f"{dataset_path}/RandAugment/sample_images"
        augmented_tensor = torch.load(f"{augmented_dir}/magnitude_{magnitude}.pt")
        save_img_grid(augmented_tensor[233], f"{sample_img_dir}/magnitude_{magnitude}.png")

    # ColorJitter
    for magnitude in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"magnitude: {magnitude}")
        augmented_dir = f"{dataset_path}/ColorJitter/augmented_tensor"
        sample_img_dir = f"{dataset_path}/ColorJitter/sample_images"
        augmented_tensor = torch.load(f"{augmented_dir}/magnitude_{magnitude}.pt")
        save_img_grid(augmented_tensor[233], f"{sample_img_dir}/magnitude_{magnitude}.png")
