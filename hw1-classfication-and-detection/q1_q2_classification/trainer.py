from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)  # (N, 20)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            # After sigmoid sigmoid(Z) = 1 / (1 + exp(-Z)) = y_pred
            # loss_matrix = - [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]  ==>  log(1 + exp(z)) - y * z
            # logsumexp trick: log(1 + exp(z)) = max(0, z) + log(1 + exp(-abs(z)))
            # Numerically stable BCE
            loss_matrix = torch.clamp(output, min=0) - output * target + torch.log1p(torch.exp(-torch.abs(output)))

            # Apply weights (ignore difficult=1)
            loss_matrix = loss_matrix * wgt

            # Average
            loss = loss_matrix.sum() / wgt.sum()
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            loss.backward()

            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print(f'Train Epoch: {epoch} [{cnt} ({100 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()

            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, mAP = utils.eval_dataset_map(model, args.device, test_loader)
                print("mAP: ", mAP)
                writer.add_scalar("mAP", mAP, cnt)
                model.train()

            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, mAP = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, mAP
