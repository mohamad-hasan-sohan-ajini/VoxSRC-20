import torch


def save_checkpoint(model, criterion, optimizer, scheduler, epoch):
    torch.save(
        model.state_dict(),
        f'checkpoints/model_{epoch+1:05d}.pt'
    )
    if criterion:
        torch.save(
            criterion.state_dict(),
            f'checkpoints/criterion_{epoch+1:05d}.pt'
        )
    if optimizer:
        torch.save(
            optimizer.state_dict(),
            f'checkpoints/optimizer_{epoch+1:05d}.pt'
        )
    if scheduler:
        torch.save(
            scheduler.state_dict(),
            f'checkpoints/scheduler_{epoch+1:05d}.pt'
        )


def load_checkpoint(model, model_path, device):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
