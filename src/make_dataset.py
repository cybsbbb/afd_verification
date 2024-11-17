import os
import cv2
import torch
from pathlib import Path
import albumentations as A
import torchvision.transforms as transforms
from torchvision.transforms import v2
from plot_utils import *

script_path = Path(__file__).resolve().parent


def get_image_paths():
    # select 1000 images from iamgenet-mini
    paths = []
    labels = []
    print(f'{script_path}/../data/imagenet-mini/imagenet-mini/train')
    for dirname, _, filenames in os.walk(f'{script_path}/../data/imagenet-mini/train'):
        for filename in filenames:
            if filename[-4:] == 'JPEG':
                paths += [(os.path.join(dirname, filename))]
                label = dirname.split('/')[-1]
                labels += [label]
    # Get the 1000 image path
    final_paths = []
    final_labels = []
    label_seen = set()
    for path, label in zip(paths, labels):
        if label not in label_seen:
            final_paths += [path]
            final_labels += [label]
            label_seen.add(label)
    return final_paths


def construct_rand_augment_tensor(final_paths, magnitude, num_ops=2):
    # magnitude with in [0, 31]
    ans = []
    # Declare an augmentation pipeline
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
    ])
    augment_transform = v2.Compose([
        v2.RandAugment(num_ops=num_ops, magnitude=magnitude),
        v2.ToTensor(),
    ])
    for idx, image_path in enumerate(final_paths[:]):
        if (idx + 1) % 100 == 0:
            print(idx)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        single_augments = []
        for _ in range(16):
            augmented_image = augment_transform(image)
            single_augments.append(augmented_image)
        single_augments_tensor = torch.stack(single_augments, dim=0)
        ans.append(single_augments_tensor)
    final_tensor = torch.stack(ans, dim=0)
    print(final_tensor.shape)
    return final_tensor


def construct_color_jitter_augment_tensor(final_paths, magnitude):
    # magnitude with in [0, 0.5]
    ans = []
    # Declare an augmentation pipeline
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
    ])
    augment_transform = v2.Compose([
        v2.ColorJitter(brightness=magnitude, contrast=magnitude, saturation=magnitude, hue=magnitude/2),
        v2.ToTensor(),
    ])
    for idx, image_path in enumerate(final_paths[:]):
        if (idx + 1) % 100 == 0:
            print(idx)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        single_augments = []
        for _ in range(16):
            augmented_image = augment_transform(image)
            single_augments.append(augmented_image)
        single_augments_tensor = torch.stack(single_augments, dim=0)
        ans.append(single_augments_tensor)
    final_tensor = torch.stack(ans, dim=0)
    print(final_tensor.shape)
    return final_tensor


if __name__ == '__main__':
    dataset_path = f'{script_path}/../data'
    final_paths = get_image_paths()

    # Rand Augmentation
    # for magnitude in [1, 5, 9, 13, 17, 21]:
    #     print(f"magnitude: {magnitude}")
    #     augmented_tensor = construct_rand_augment_tensor(final_paths, magnitude=magnitude)
    #     augmented_dir = f"{dataset_path}/RandAugment/augmented_tensor"
    #     if not os.path.exists(augmented_dir):
    #         os.makedirs(augmented_dir)
    #     sample_img_dir = f"{dataset_path}/RandAugment/sample_images"
    #     if not os.path.exists(sample_img_dir):
    #         os.makedirs(sample_img_dir)
    #     save_img_grid(augmented_tensor[233], f"{sample_img_dir}/magnitude_{magnitude}.png")
    #     torch.save(augmented_tensor, f"{augmented_dir}/magnitude_{magnitude}.pt")

    # ColorJitter
    for magnitude in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"magnitude: {magnitude}")
        augmented_tensor = construct_color_jitter_augment_tensor(final_paths, magnitude=magnitude)
        augmented_dir = f"{dataset_path}/ColorJitter/augmented_tensor"
        if not os.path.exists(augmented_dir):
            os.makedirs(augmented_dir)
        sample_img_dir = f"{dataset_path}/ColorJitter/sample_images"
        if not os.path.exists(sample_img_dir):
            os.makedirs(sample_img_dir)
        save_img_grid(augmented_tensor[233], f"{sample_img_dir}/magnitude_{magnitude}.png")
        torch.save(augmented_tensor, f"{augmented_dir}/magnitude_{magnitude}.pt")
