import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # Original GAN discriminator loss: -E[log(D(x))] - E[log(1 - D(G(z)))]
    # We want to maximize log(D(x)) + log(1 - D(G(z)))
    # So we minimize the negative: minimize -log(D(x)) - log(1 - D(G(z)))
    # Using BCE: -log(sigmoid(x)) for real (label 1), -log(1-sigmoid(x)) for fake (label 0)
    # Since discriminator outputs logits, we use binary cross entropy with logits
    # ref: https://neptune.ai/blog/gan-loss-functions
    loss_real = F.binary_cross_entropy_with_logits(
        discrim_real, torch.ones_like(discrim_real)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        discrim_fake, torch.zeros_like(discrim_fake)
    )
    loss = loss_real + loss_fake
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # Original GAN generator loss: minimize -E[log(D(G(z)))]
    # Equivalently: maximize E[log(D(G(z)))]
    # We want discriminator to output high values (close to 1) for fake images
    # So we use BCE with target = 1
    loss = F.binary_cross_entropy_with_logits(
        discrim_fake, torch.ones_like(discrim_fake)
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(20000),  # Early stopping for preventing FID explosion
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
