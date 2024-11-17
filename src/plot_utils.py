import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image


# Use make_grid to create a grid of images
def show_img_grid(sample, nrow=4, ncol=4):
    sample = (sample * 255).clamp(0, 255).to(th.uint8)
    sample = sample.contiguous()

    # Ensure the number of images matches the grid dimensions
    total_images = nrow * ncol
    sample = sample[:total_images]  # Adjust sample size to fit the grid

    # Create the grid with specified number of rows
    grid = vutils.make_grid(sample, nrow=nrow, padding=2, normalize=False)

    # Convert the grid to a numpy array
    grid_np = grid.cpu().numpy()

    # Transpose the numpy array from (C, H, W) to (H, W, C) for displaying
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Display the grid of images
    plt.figure(figsize=(ncol, nrow), dpi=300)  # Adjust size based on grid dimensions
    plt.imshow(grid_np)
    plt.axis('off')  # Turn off the axis numbers and ticks
    plt.show()
    return 0


def save_img_grid(sample, save_path, nrow=4, ncol=4):
    sample = (sample * 255).clamp(0, 255).to(th.uint8)
    sample = sample.contiguous()

    # Ensure the number of images matches the grid dimensions
    total_images = nrow * ncol
    sample = sample[:total_images]  # Adjust sample size to fit the grid

    # Create the grid with specified number of rows
    grid = vutils.make_grid(sample, nrow=nrow, padding=2, normalize=False)

    # Convert the grid to a numpy array
    grid_np = grid.cpu().numpy()

    # Transpose the numpy array from (C, H, W) to (H, W, C) for saving
    grid_np = np.transpose(grid_np, (1, 2, 0))

    # Save the grid of images
    Image.fromarray(grid_np).save(save_path)
