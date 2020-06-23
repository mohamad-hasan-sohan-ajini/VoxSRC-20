import os

import torch


def save_checkpoint(save_path, model, criterion, optimizer, epoch):
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f'model_{epoch+1:05d}.pt')
    )
    if criterion:
        torch.save(
            criterion.state_dict(),
            os.path.join(save_path, f'criterion_{epoch+1:05d}.pt')
        )
    if optimizer:
        torch.save(
            optimizer.state_dict(),
            os.path.join(save_path, f'optimizer_{epoch+1:05d}.pt')
        )


def load_checkpoint(model, model_path, device):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
