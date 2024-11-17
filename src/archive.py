import os
import cv2
import torch
from pathlib import Path
import albumentations as A
import torchvision.transforms as transforms
from.make_dataset import get_image_paths
from plot_utils import *

script_path = Path(__file__).resolve().parent


def augmentation_colorjitter(image_path, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.SmallestMaxSize(max_size=256, p=1),
        A.CenterCrop(height=256, width=256, p=1),
        A.ColorJitter(brightness=brightness,
                      contrast=contrast,
                      saturation=saturation,
                      hue=hue, p=1),
    ])
    transformed_image = transform(image=image)['image']
    return transformed_image


def construct_original_tensor(final_paths):
    ans = []
    trans = transforms.ToTensor()
    for image_path in final_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Declare an augmentation pipeline, without the ColorJitter
        transform = A.Compose([
            A.SmallestMaxSize(max_size=256, p=1),
            A.CenterCrop(height=256, width=256, p=1),
        ])
        transformed_image = transform(image=image)['image']
        image_tensor = trans(transformed_image)
        ans.append(image_tensor)
    final_tensor = torch.stack(ans, dim=0)
    print(final_tensor.shape)
    return final_tensor


def construct_transformed_tensor(final_paths, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    ans = []
    trans = transforms.ToTensor()
    for image_path in final_paths:
        transformed_image = augmentation_colorjitter(image_path,
                                                     brightness=brightness,
                                                     contrast=contrast,
                                                     saturation=saturation,
                                                     hue=hue)
        image_tensor = trans(transformed_image)
        ans.append(image_tensor)
    final_tensor = torch.stack(ans, dim=0)
    print(final_tensor.shape)
    return final_tensor


if __name__ == '__main__':
    dataset_path = f'{script_path}/../data'
    final_paths = get_image_paths()

    # no use now
    original_tensor = construct_original_tensor(final_paths)
    result_dir = f"{dataset_path}/original"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img_grid(original_tensor, f"{result_dir}/samples.png")
    torch.save(original_tensor, f"{result_dir}/tensor.pt")

    # no use now
    for brightness in [0.0, 0.2, 0.4, 0.6]:
        for contrast in [0.0, 0.2, 0.4, 0.6]:
            for saturation in [0.0, 0.2, 0.4, 0.6]:
                for hue in [0.0, 0.1, 0.3, 0.5]:
                    augmented_tensor = construct_transformed_tensor(final_paths, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
                    augmented_dir = f"{dataset_path}/b{brightness}_c{contrast}_s{saturation}_h{hue}"
                    if not os.path.exists(augmented_dir):
                        os.makedirs(augmented_dir)
                    save_img_grid(augmented_tensor, f"{augmented_dir}/samples.png")
                    torch.save(augmented_tensor, f"{augmented_dir}/tensor.pt")
