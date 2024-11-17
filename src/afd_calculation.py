import os
import k_diffusion as K
import torch as th
import numpy as np
import torch
import csv
from pathlib import Path
from plot_utils import save_img_grid
from torch.nn.functional import pdist
from tqdm import tqdm


script_path = Path(__file__).resolve().parent


def compute_feature(extractor, imgs, n, batch_size):
    features_list = []
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        features = extractor(imgs[start_idx: end_idx])
        features = features.detach().cpu()
        features_list.append(features)
    return th.cat(features_list)


def compute_afd(augmented_tensor, device='cpu'):
    batch_size = 16
    extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
    sample_num = augmented_tensor.shape[0]
    augment_num = augmented_tensor.shape[1]
    diversities = []
    for i in range(sample_num):
        single_sample_feature = compute_feature(extractor, augmented_tensor[i], augment_num, batch_size)
        distances = pdist(single_sample_feature, p=2)
        average_distance = distances.mean().item()
        diversities.append(average_distance)
    afd = np.mean(np.array(diversities))
    print(afd)
    return afd


if __name__ == "__main__":
    device = 'cuda'
    dataset_path = f'{script_path}/../data'
    with open(f'{dataset_path}/result_table.csv', 'w', newline='') as csvfile:
        fieldnames = ['Augment / magnitude'] + [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        row = dict()
        row['Augment / magnitude'] = 'ColorJitter'
        for magnitude in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            augmented_dir = f"{dataset_path}/ColorJitter/augmented_tensor"
            augmented_tensor = torch.load(f"{augmented_dir}/magnitude_{magnitude}.pt", map_location=device)
            afd = compute_afd(augmented_tensor, device)
            row[magnitude] = afd
            print(magnitude, afd)
            del augmented_tensor
            torch.cuda.empty_cache()
        writer.writerow(row)
