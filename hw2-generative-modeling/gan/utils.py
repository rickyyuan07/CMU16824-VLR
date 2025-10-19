import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    def gen_fn(z): return (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    import torch
    from torchvision.utils import save_image

    # Create 10x10 grid interpolating first two dimensions
    steps = 10
    z_samples = []

    # Linearly interpolate from -1 to 1 for both first two dimensions
    for i in range(steps):
        for j in range(steps):
            z = torch.zeros(128)
            z[0] = -1 + 2 * i / (steps - 1)  # First dim: -1 to 1
            z[1] = -1 + 2 * j / (steps - 1)  # Second dim: -1 to 1
            # Rest of dimensions stay at 0
            z_samples.append(z)

    # Stack all samples into a batch
    z_batch = torch.stack(z_samples).to(next(gen.parameters()).device)

    # Generate images
    generated_images = gen.forward_given_samples(z_batch)

    # Rescale from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2

    # Save the grid of images
    save_image(generated_images, path, nrow=steps)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
