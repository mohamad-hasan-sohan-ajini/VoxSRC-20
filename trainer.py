import os
import argparse
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opts import create_argparser
from data_loader import VoxCelebDataset
from model import UniversalSRModel
from loss import CosFace, PSGE2E, Prototypical
from utils import save_checkpoint, load_checkpoint
from evaluation import EER_metric

# add argparser functions
args = create_argparser().parse_args()
kwargs = vars(args)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# data loader
ds = VoxCelebDataset(
    args.sample_rate,
    args.win_length,
    args.hop_length,
    args.n_frames,
    args.n_fft,
    args.n_filterbanks,
    args.feat_type,
    'dev',
    args.dev_csv,
    args.samples_per_speaker
)
dl = DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)
args.num_spkr = len(ds)

# model
if args.feat_type == 'mel':
    n_feat = args.n_filterbanks
elif args.feat_type == 'spect':
    n_feat = args.n_fft // 2 + 1
model = UniversalSRModel(n_feat, **kwargs)
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
optimizer.zero_grad()

# lr schedule
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.scheduler_step_size,
    gamma=args.scheduler_gamma
)
load_checkpoint(scheduler, args.scheduler_path, device)

# training loop
counter = args.start_epoch * len(dl)
for epoch in range(args.start_epoch, args.num_epochs):
    print('-' * 20 + f'epoch: {epoch+1:05d}' + '-' * 20)
    for x, target in tqdm(dl):
        x = x.view(-1, 1, n_feat, args.n_frames).to(device)
        target = target.view(-1, 1).to(device)

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

        loss.backward()
        if (counter + 1) % args.update_interleaf == 0:
            optimizer.step()
            optimizer.zero_grad()

        # log the loss value
        log.add_scalar('train-loss', loss.item(), counter)
        counter += 1

    scheduler.step()
    if (epoch + 1) % args.test_interleaf == 0:
        eer = EER_metric(model, device, args)
        log.add_scalar('test-EER', eer, epoch + 1)

        if args.save_checkpoint:
            save_checkpoint(args.save_path, model, criterion, optimizer, epoch)

save_checkpoint(args.save_path, model, criterion, optimizer, epoch)
