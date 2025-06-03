import os
import time
import yaml
import random
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
np.int = int   # ← add this
from typing import Optional, Tuple

import torchvision
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.distributed

import argparse
from thop import profile

from models.submodules.layers import Conv1x1, Conv3x3, Linear, SpikingMatmul
import models.spikingresformer
from utils.augment import DVSAugment
from utils.scheduler import BaseSchedulerPerEpoch, BaseSchedulerPerIter
from utils.utils import RecordDict, GlobalTimer, Timer
from utils.utils import count_convNd, count_linear, count_matmul
from utils.utils import DatasetSplitter, DatasetWarpper, CriterionWarpper, DVStransform, SOPMonitor
from utils.utils import is_main_process, save_on_master, tb_record, accuracy, safe_makedirs, tb_record_triplet
from spikingjelly.activation_based import functional, layer, base
from timm.data import FastCollateMixup, create_loader
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.models import create_model
from utils.triplet_face import TripletFaceDataset
from utils.tripletloss import TripletLoss

import math


def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)

    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description='Training')

    # training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='spikingresformer_ti', help='model type')
    # parser.add_argument('--dataset', default='ImageNet', help='dataset type')
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')

    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[])
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # triplet load
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name: imagenet or tripletface')
    parser.add_argument('--triplet-list-train', type=str, default='',
                        help='Path to training triplet list txt file')
    parser.add_argument('--triplet-list-val',   type=str, default='',
                        help='Path to validation triplet list txt file')


    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, embeddings, labels):
        # Handle different input shapes
        if embeddings.dim() == 3:  # [T, B, embed_dim] - from TET
            # Take mean across time steps
            embeddings = embeddings.mean(0)  # [B, embed_dim]
        elif embeddings.dim() == 2:  # [B, embed_dim] - normal case
            pass
        else:
            raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")
        
        # # Debug prints
        # print(f"ArcFace input embeddings shape: {embeddings.shape}")
        # print(f"ArcFace labels shape: {labels.shape}")
        # print(f"Weight shape: {self.weight.shape}")
        
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [B, embed_dim]
        weight_norm = F.normalize(self.weight, p=2, dim=1)     # [num_classes, embed_dim]
        
        # Compute cosine similarity
        cos_theta = F.linear(embeddings_norm, weight_norm)  # [B, num_classes]
        
        # Compute sin_theta
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-7)
        
        # Apply margin
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Create mask for target classes
        mask = torch.zeros_like(cos_theta).scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Apply margin only to correct class
        output = cos_theta * (1.0 - mask) + cos_theta_m * mask
        output = output * self.s
        
        # Compute cross entropy loss
        loss = F.cross_entropy(output, labels)
        return loss


def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger


def init_distributed(logger: logging.Logger, distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        logger.info('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    logger.info('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    # only master process logs
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank


class TETArcFaceLoss(nn.Module):
    """TET-aware ArcFace loss that handles temporal outputs properly"""
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.5, TET_phi=1.0, TET_lambda=0.0):
        super().__init__()
        self.arcface = ArcFaceLoss(embed_dim, num_classes, s, m)
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        
    def forward(self, embeddings, labels):
        if embeddings.dim() == 3:  # [T, B, embed_dim]
            T, B, D = embeddings.shape
            total_loss = 0.0
            
            # Apply ArcFace loss at each time step
            for t in range(T):
                step_loss = self.arcface(embeddings[t], labels)  # [B, embed_dim]
                total_loss += (1.0 - self.TET_lambda) * step_loss
            
            # Add TET regularization if needed
            if self.TET_phi > 0:
                # Mean squared error between consecutive time steps
                mse_loss = 0.0
                for t in range(1, T):
                    mse_loss += F.mse_loss(embeddings[t], embeddings[t-1])
                total_loss += self.TET_phi * self.TET_lambda * mse_loss / (T - 1)
            
            return total_loss / T
        else:
            # Standard case
            return self.arcface(embeddings, labels)


def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    dataset_type: str,
    input_size: Tuple[int],
    distributed: bool,
    augment: str,
    mixup: bool,
    cutout: bool,
    label_smoothing: float,
    T: int,
    triplet_list_train: str,
    triplet_list_val: str,
):
    if dataset_type.lower() == 'tripletface':
        # Common transforms
        transform = transforms.Compose([
            transforms.Resize(input_size[-2:]),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Training set with identity labels
        train_ds = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=transform
        )
        # Validation set remains triplet-based
        test_ds = TripletFaceDataset(triplet_list_val, transform=transform)
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_ds)
            test_sampler = torch.utils.data.SequentialSampler(test_ds)
        
        data_loader_train = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=workers, pin_memory=True, drop_last=True,
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, sampler=test_sampler,
            num_workers=workers, pin_memory=True, drop_last=False,
        )
        return train_ds, test_ds, data_loader_train, data_loader_test


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_train: torch.utils.data.DataLoader,
    logger: logging.Logger,
    print_freq: int,
    factor: int,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
):
    model.train()
    metric_dict = RecordDict({'loss': None})
    timer_container = [0.0]
    model.zero_grad()
    
    for idx, (images, labels) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            images, labels = images.cuda(), labels.cuda()
            
            if scaler is not None:
                with autocast():
                    embeddings = model(images)  # Shape: [T, B, embed_dim] or [B, embed_dim]
                    # print(f"Raw embeddings shape: {embeddings.shape}")
                    
                    # Let ArcFace handle the dimension reduction
                    loss = criterion(embeddings, labels)
            else:
                embeddings = model(images)
                # print(f"Raw embeddings shape: {embeddings.shape}")
                loss = criterion(embeddings, labels)
            
            metric_dict['loss'].update(loss.item())
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                model.zero_grad()
            
            if scheduler_per_iter is not None:
                scheduler_per_iter.step()
            
            functional.reset_net(model)
        
        batch_size = images.size(0)
        if print_freq and ((idx + 1) % int(len(data_loader_train) / print_freq) == 0):
            metric_dict.sync()
            logger.debug(
                f' [{idx+1}/{len(data_loader_train)}] it/s: '
                f'{(idx+1)*batch_size*factor/timer_container[0]:.5f}, '
                f'loss: {metric_dict["loss"].ave:.5f}'
            )
    
    metric_dict.sync()
    return metric_dict['loss'].ave

def train_one_epoch_triplet(
    model, criterion, optimizer,
    data_loader_train, logger,
    print_freq, factor,
    scheduler_per_iter=None, scaler=None,
):
    model.train()
    metric_dict = RecordDict({'loss': None})
    timer_container = [0.0]
    model.zero_grad()

    for idx, (anchor, positive, negative) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            # Move each tensor to GPU
            anchor   = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

            # Concatenate into one big batch: [3*B, C, H, W]
            images = torch.cat([anchor, positive, negative], dim=0)

            # Forward + loss
            if scaler is not None:
                with autocast():
                    embeddings = model(images)
                    loss = criterion(embeddings)
            else:
                embeddings = model(images)
                loss = criterion(embeddings)

            metric_dict['loss'].update(loss.item())

            # Backward + step
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                model.zero_grad()

            # Iter‐level scheduler
            if scheduler_per_iter is not None:
                scheduler_per_iter.step()

            functional.reset_net(model)

        # Logging
        batch_size = anchor.size(0)
        if print_freq and ((idx + 1) % int(len(data_loader_train) / print_freq) == 0):
            metric_dict.sync()
            logger.debug(
                f' [{idx+1}/{len(data_loader_train)}] it/s: '
                f'{(idx+1)*batch_size*factor/timer_container[0]:.5f}, '
                f'loss: {metric_dict["loss"].ave:.5f}'
            )

    metric_dict.sync()
    return metric_dict['loss'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, one_hot=None):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(), target.cuda()
            if one_hot:
                target = F.one_hot(target, one_hot).float()
            output = model(image)
            loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            if target.dim() > 1:
                target = target.argmax(-1)
            acc1, acc5 = accuracy(output.mean(0), target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % int(len(data_loader) / print_freq)) == 0:
                #torch.distributed.barrier()
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    #torch.distributed.barrier()
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

def evaluate_triplet(model, data_loader, print_freq, logger):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'accuracy': None})
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(data_loader):
            # Move to GPU
            anchor   = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

            # Concatenate into a single batch: [3*B, C, H, W]
            images = torch.cat([anchor, positive, negative], dim=0)

            # Forward pass: embeddings shape [T, 3*B, embed_dim]
            embeddings = model(images)

            # Reshape to [T, B, 3, embed_dim]
            T, total, embed_dim = embeddings.shape
            B = total // 3
            embeddings = embeddings.reshape(T, B, 3, embed_dim)

            # Split out the triplet views
            anchor_e   = embeddings[:, :, 0]  # [T, B, embed_dim]
            positive_e = embeddings[:, :, 1]
            negative_e = embeddings[:, :, 2]

            # Compute squared distances [T, B]
            pos_dist = torch.sum((anchor_e - positive_e) ** 2, dim=-1)
            neg_dist = torch.sum((anchor_e - negative_e) ** 2, dim=-1)

            # Triplet loss: margin=0.2
            loss = torch.clamp(pos_dist - neg_dist + 0.2, min=0).mean()
            metric_dict['loss'].update(loss.item())

            # Accuracy: fraction of (pos_dist < neg_dist)
            correct = (pos_dist < neg_dist).float().mean()
            metric_dict['accuracy'].update(correct.item(), B)

            # reset spiking states
            functional.reset_net(model)

            # Logging
            if print_freq and ((idx + 1) % int(len(data_loader) / print_freq) == 0):
                metric_dict.sync()
                logger.debug(
                    f' [{idx+1}/{len(data_loader)}] '
                    f'loss: {metric_dict["loss"].ave:.5f}, '
                    f'accuracy: {metric_dict["accuracy"].ave:.5f}'
                )

    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['accuracy'].ave


def test(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    input_size: Tuple[int],
    args: argparse.Namespace,
    logger: logging.Logger,
):

    logger.info('[Test]')
    mon = SOPMonitor(model)
    model.eval()
    mon.enable()
    logger.debug('Test start')
    metric_dict = RecordDict({'acc@1': None, 'acc@5': None}, test=True)
    with torch.no_grad():
        t = time.time()
        for idx, (image, target) in enumerate(data_loader_test):
            image, target = image.cuda(), target.cuda()
            output = model(image).mean(0)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if args.print_freq != 0 and ((idx + 1) %
                                         int(len(data_loader_test) / args.print_freq)) == 0:
                logger.debug('Test: [{}/{}]'.format(idx + 1, len(data_loader_test)))
        logger.info('Throughput: {:.5f} it/s'.format(
            len(data_loader_test) * args.batch_size / (time.time() - t)))

    metric_dict.sync()
    logger.info('Acc@1: {:.5f}, Acc@5: {:.5f}'.format(metric_dict['acc@1'].ave,
                                                      metric_dict['acc@5'].ave))

    step_mode = 's'
    for m in model.modules():
        if isinstance(m, base.StepModule):
            if m.step_mode == 'm':
                step_mode = 'm'
            else:
                step_mode = 's'
            break

    ops, params = profile(
        model, inputs=(torch.rand(input_size).cuda().unsqueeze(0), ), verbose=False, custom_ops={
            layer.Conv2d: count_convNd,
            Conv3x3: count_convNd,
            Conv1x1: count_convNd,
            Linear: count_linear,
            SpikingMatmul: count_matmul, })[0:2]
    if step_mode == 'm':
        ops, params = (ops / (1000**3)) / args.T, params / (1000**2)
    else:
        ops, params = (ops / (1000**3)), params / (1000**2)
    functional.reset_net(model)
    logger.info('MACs: {:.5f} G, params: {:.2f} M.'.format(ops, params))

    sops = 0
    for name in mon.monitored_layers:
        sublist = mon[name]
        sop = torch.cat(sublist).mean().item()
        sops = sops + sop
    sops = sops / (1000**3)
    # input is [N, C, H, W] or [T*N, C, H, W]
    sops = sops / args.batch_size
    if step_mode == 's':
        sops = sops * args.T
    logger.info('Avg SOPs: {:.5f} G, Power: {:.5f} mJ.'.format(sops, 0.9 * sops))
    logger.info('A/S Power Ratio: {:.6f}'.format((4.6 * ops) / (0.9 * sops)))

def test_triplet(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    print_freq: int,
    logger: logging.Logger,
    margin: float = 0.2,
):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'accuracy': None})
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(data_loader_test):
            # Move all to GPU
            anchor   = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

            # Build a single batch
            images = torch.cat([anchor, positive, negative], dim=0)
            embeddings = model(images)  # [T, 3*B, D]

            # Reshape to separate triplets
            T, total, D = embeddings.shape
            B = total // 3
            embeddings = embeddings.view(T, B, 3, D)
            a_e = embeddings[:, :, 0]
            p_e = embeddings[:, :, 1]
            n_e = embeddings[:, :, 2]

            # Compute distances
            pos_dist = torch.sum((a_e - p_e) ** 2, dim=-1)  # [T, B]
            neg_dist = torch.sum((a_e - n_e) ** 2, dim=-1)  # [T, B]

            # Triplet loss + accuracy
            loss = torch.clamp(pos_dist - neg_dist + margin, min=0).mean()
            correct = (pos_dist < neg_dist).float().mean()

            metric_dict['loss'].update(loss.item())
            metric_dict['accuracy'].update(correct.item(), B)

            functional.reset_net(model)

            # Logging
            if print_freq and ((idx + 1) % int(len(data_loader_test) / print_freq) == 0):
                metric_dict.sync()
                logger.debug(
                    f' [{idx+1}/{len(data_loader_test)}] '
                    f'loss: {metric_dict["loss"].ave:.5f}, '
                    f'accuracy: {metric_dict["accuracy"].ave:.5f}'
                )

    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['accuracy'].ave


def main():
    ##################################################
    #                       setup
    ##################################################

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # load data
    dataset_type = args.dataset
    one_hot = None
    if dataset_type == 'CIFAR10':
        input_size = (3, 32, 32)
    # ... (other dataset types)
    elif dataset_type.lower() == 'tripletface':
        input_size = (3, 224, 224)
        embed_dim = 512  # Embedding dimension
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size

    # Call load_data without num_classes
    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers, dataset_type, input_size,
        distributed, args.augment, args.mixup, args.cutout, args.label_smoothing, args.T,
        args.triplet_list_train, args.triplet_list_val)

    # Set num_classes for tripletface after loading dataset
    num_classes = None  # Default
    if dataset_type.lower() == 'tripletface':
        num_classes = len(dataset_train.classes)  # Number of identities
    elif dataset_type == 'CIFAR10':
        num_classes = 10
    else:
        raise ValueError(f"num_classes not set for dataset_type {dataset_type}")

    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    # Model
    model = create_model(
        args.model,
        T=args.T,
        num_classes=embed_dim,  # Output embeddings, not logits
        img_size=input_size[-1],
    ).cuda()

    # Transfer
    if args.transfer:
        checkpoint = torch.load(args.transfer, map_location='cpu')
        model.transfer(checkpoint['model'])

    # Optimizer
    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Loss function
    if dataset_type.lower() == 'tripletface':
        criterion = ArcFaceLoss(embed_dim=embed_dim, num_classes=num_classes).cuda()  # <- Add .cuda()
        criterion_eval = TripletLoss(margin=0.2)
    else:
        margin = 0.2
        criterion = TripletLoss(margin=margin)
        criterion_eval = TripletLoss(margin=margin)

    if args.TET:
        criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
        criterion_eval = CriterionWarpper(criterion_eval)

    # AMP speed up
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # LR scheduler
    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-5,
        warmup_lr=1e-5,
        warmup_epochs=3,
    )

    # Sync BN
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if distributed and not args.test_only:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                         find_unused_parameters=False)
        model_without_ddp = model.module

    # Custom scheduler
    scheduler_per_iter = None
    scheduler_per_epoch = None

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        max_acc1 = checkpoint['max_acc1']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info('Resume from epoch {}'.format(start_epoch))
        start_epoch += 1
    else:
        start_epoch = 0
        max_acc1 = 0

    logger.debug(str(model))

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        if distributed:
            logger.error('Using distribute mode in test, abort')
            return
        test_triplet(model_without_ddp, data_loader_test, args.print_freq, logger)
        return

    ##################################################
    #                   Train
    ##################################################

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'),
                                  purge_step=start_epoch)

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))

        with Timer(' Train', logger):
            # Use train_one_epoch for ArcFace-based training
            train_loss = train_one_epoch(model, criterion, optimizer,
                                         data_loader_train, logger,
                                         args.print_freq, world_size,
                                         scheduler_per_iter, scaler)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()

        with Timer(' Test', logger):
            test_loss, test_acc = test_triplet(model, data_loader_test, args.print_freq, logger)

        if is_main_process() and tb_writer is not None:
            tb_record_triplet(tb_writer, train_loss, test_loss, test_acc, epoch)

        logger.info(' Test loss: {:.5f}, Accuracy: {:.5f}'.format(test_loss, test_acc))

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1,
        }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        # Save best checkpoint
        if test_acc > max_acc1:
            max_acc1 = test_acc
            best_ckpt = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_acc1': max_acc1,
            }
            if lr_scheduler is not None:
                best_ckpt['lr_scheduler'] = lr_scheduler.state_dict()
            save_on_master(
                best_ckpt,
                os.path.join(args.output_dir, 'checkpoint_max_acc1.pth')
            )

        # Optional: save latest checkpoint
        if args.save_latest:
            latest_ckpt = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_acc1': max_acc1,
            }
            if lr_scheduler is not None:
                latest_ckpt['lr_scheduler'] = lr_scheduler.state_dict()
            save_on_master(
                latest_ckpt,
                os.path.join(args.output_dir, 'checkpoint_latest.pth')
            )

    logger.info('Training completed.')

    ##################################################
    #                   test
    ##################################################

    # Reset model
    del model, model_without_ddp

    model = create_model(
        args.model,
        T=args.T,
        num_classes=embed_dim,
        img_size=input_size[-1],
    )

    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'),
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return

    # Reload data
    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers,
        dataset_type, input_size, False,  # Removed num_classes
        args.augment, args.mixup, args.cutout,
        args.label_smoothing, args.T, 
        args.triplet_list_train, args.triplet_list_val
    )

    # Test
    if is_main_process():
        test_triplet(model.cuda(), data_loader_test, args.print_freq, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()
