import os
import argparse
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opts import add_args
from data_loader import ClassificationVCDS, MetricLearningVCDS, transform
from model import UniversalSRModel
from loss import CosFace, PSGE2E, Prototypical
from utils import save_checkpoint, load_checkpoint
from evaluation import EER_metric

# add argparser functions
parser = argparse.ArgumentParser(description='Training options')
parser = add_args(parser)
args = parser.parse_args()
kwargs = vars(args)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# data loader
if args.criterion_type == 'classification':
    ds = ClassificationVCDS(
        args.dev_csv,
        args.win_length,
        args.hop_length,
        args.num_frames
    )
    args.num_spkr = len(ds)
elif args.criterion_type == 'metriclearning':
    ds = MetricLearningVCDS(
        args.dev_csv,
        args.win_length,
        args.hop_length,
        args.num_frames,
        args.spk_samples
    )
else:
    raise ValueError('args.criterion-type: no valid criterion type')
dl = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
feature_extractor = transform(**kwargs).to(device)

# model
model = UniversalSRModel(**kwargs)
model.to(device)
load_checkpoint(model, args.model_path, device)

# log
log = SummaryWriter(args.logdir)

# criterion
if args.criterion == 'cosface':
    criterion = CosFace(args.repr_dim, args.num_spkr, args.m, args.s)
elif args.criterion == 'psge2e':
    criterion = PSGE2E(args.repr_dim, args.num_spkr, args.init_m, args.init_s)
elif args.criterion == 'prototypical':
    criterion = Prototypical(args.repr_dim, args.num_spkr)
else:
    raise ValueError('args.criterion: no valid criterion function')
criterion = criterion.to(device)
load_checkpoint(criterion, args.criterion_path, device)

# optimizer
optimizer = torch.optim.Adam(
    [
        {'params': model.parameters(), 'lr': args.lr},
        {'params': criterion.parameters(), 'lr': args.criterion_lr}
    ]
)
load_checkpoint(optimizer, args.optimizer_path, device)

# lr schedule
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.step_size,
    gamma=args.gamma
)
load_checkpoint(criterion, args.scheduler_path, device)

# training loop
counter = 0
for epoch in range(args.num_epochs):
    print('-' * 20 + f'epoch: {epoch+1:03d}' + '-' * 20)
    for x, target in tqdm(dl):
        x = x.to(device)
        x = feature_extractor(x) + 1
        x = x.log()
        target = target.to(device)

        # forward pass
        y = model(x)

        if args.criterion_type == 'classification':
            scores, loss = criterion(y, target)
            # log the accuracy
            preds = scores.topk(1, dim=1)[1]
            log.add_scalar(
                'train-acc',
                (preds == target).sum().item() / y.size(0),
                counter
            )
        elif args.criterion_type == 'metriclearning':
            # TODO: implement metriclearning methods
            pass

        # updata weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the loss value
        log.add_scalar('train-loss', loss.item(), counter)
        counter += 1

    scheduler.step()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    log.add_scalar('train-lr', lr, epoch + 1)

    if (epoch + 1) % args.test_interleaf == 0:
        eer = EER_metric(
            model,
            feature_extractor,
            args.num_frames,
            args.criterion,
            device,
            args.test_csv
        )
        log.add_scalar('test-EER', eer, epoch + 1)

        if args.save_checkpoint:
            save_checkpoint(model, criterion, optimizer, scheduler, epoch)

save_checkpoint(model, criterion, optimizer, scheduler, epoch)
