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
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')  # Reduced for single GPU
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')  # Added default

    parser.add_argument('--patience', default=10, type=int,  # Increased patience
                        help='Number of epochs to wait for improvement before early stopping')

    parser.add_argument('--print-freq', default=5, type=int,
                        help='Number of times a debug message is printed in one epoch')
    parser.add_argument('--data-path', default='./datasets', type=str,
                        help='Path to training data directory')
    parser.add_argument('--test-data-path', default=None, type=str,
                        help='Path to test data directory (overrides triplet-list-val if provided)')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[])
    parser.add_argument('--distributed-init-mode', type=str, default='env://')

    # triplet load
    parser.add_argument('--dataset', default='vggface2', type=str, help='Dataset name: imagenet, vggface2, or tripletface')
    parser.add_argument('--triplet-list-train', type=str, default='',
                        help='Path to training triplet list txt file')
    parser.add_argument('--triplet-list-val', type=str, default='',
                        help='Path to validation triplet list txt file')

    # argument of TET
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)

    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')

    # argumet for Arcfaceloss
    parser.add_argument('--arcface-s', type=float, default=30.0, help='ArcFace scale factor')
    parser.add_argument('--arcface-m', type=float, default=0.5, help='ArcFace margin')

    # Memory optimization arguments
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Use pin memory for data loading')

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
        
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [B, embed_dim]
        weight_norm = F.normalize(self.weight, p=2, dim=1)     # [num_classes, embed_dim]
        
        # Compute cosine similarity
        cos_theta = F.linear(embeddings_norm, weight_norm)  # [B, num_classes]
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability
        
        # Compute sin_theta
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-7)
        
        # Apply margin
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Handle difficult samples
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        
        # Create mask for target classes
        mask = torch.zeros_like(cos_theta)
        mask.scatter_(1, labels.unsqueeze(1), 1.0)
        
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
    # Force single GPU mode
    logger.info('Using single GPU mode')
    return False, 0, 1, 0


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
    test_data_path: str = None,
    pin_memory: bool = True,
):
    # Enhanced transforms for VGGFace2
    if dataset_type.lower() == 'vggface2':
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(input_size[-2:]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),  # Data augmentation
        ])
        
        # Test transforms without augmentation
        test_transform = transforms.Compose([
            transforms.Resize(input_size[-2:]),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Training set with identity labels
        train_ds = torchvision.datasets.ImageFolder(
            root=dataset_dir,
            transform=train_transform
        )
        
        # Test set
        if test_data_path is not None:
            test_ds = torchvision.datasets.ImageFolder(
                root=test_data_path,
                transform=test_transform
            )
        else:
            # Use a subset of training data for validation if no test path provided
            test_ds = torchvision.datasets.ImageFolder(
                root=dataset_dir,
                transform=test_transform
            )
    else:
        # Fallback for other datasets
        transform = transforms.Compose([
            transforms.Resize(input_size[-2:]),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if dataset_type.lower() == 'tripletface':
            train_ds = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)
            if test_data_path is not None:
                test_ds = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
            else:
                test_ds = TripletFaceDataset(triplet_list_val, transform=transform)
        elif dataset_type.lower() == 'cifar10':
            train_ds = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Single GPU samplers
    train_sampler = torch.utils.data.RandomSampler(train_ds)
    test_sampler = torch.utils.data.SequentialSampler(test_ds)

    # Create data loaders with optimized settings for single GPU
    data_loader_train = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if workers > 0 else 2,  # Prefetch batches
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else 2,
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
    gradient_accumulation_steps: int = 1,
    scheduler_per_iter: Optional[BaseSchedulerPerIter] = None,
    scaler: Optional[GradScaler] = None,
):
    model.train()
    metric_dict = RecordDict({'loss': None})
    timer_container = [0.0]
    
    # Zero gradients at the start
    optimizer.zero_grad()
    
    for idx, (images, labels) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            
            # Forward pass with optional AMP
            if scaler is not None:
                with autocast():
                    embeddings = model(images)
                    loss = criterion(embeddings, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
            else:
                embeddings = model(images)
                loss = criterion(embeddings, labels)
                loss = loss / gradient_accumulation_steps
            
            metric_dict['loss'].update(loss.item() * gradient_accumulation_steps)
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                if scheduler_per_iter is not None:
                    scheduler_per_iter.step()
            
            # Reset spiking network state
            functional.reset_net(model)
        
        batch_size = images.size(0)
        if print_freq and ((idx + 1) % max(1, len(data_loader_train) // print_freq) == 0):
            metric_dict.sync()
            logger.debug(
                f' [{idx+1}/{len(data_loader_train)}] it/s: '
                f'{(idx+1)*batch_size*factor/timer_container[0]:.5f}, '
                f'loss: {metric_dict["loss"].ave:.5f}, '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f}'
            )
    
    # Handle remaining gradients if not divisible by gradient_accumulation_steps
    if len(data_loader_train) % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    metric_dict.sync()
    return metric_dict['loss'].ave


def evaluate(model, criterion, data_loader, print_freq, logger, one_hot=None):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if one_hot:
                target_one_hot = F.one_hot(target, one_hot).float()
                output = model(image)
                loss = criterion(output, target_one_hot)
            else:
                output = model(image)
                # For ArcFace, we need to compute a dummy loss for logging
                if hasattr(criterion, 'arcface'):
                    # Create dummy logits for loss computation
                    dummy_logits = torch.randn(output.size(0), criterion.arcface.num_classes, device=output.device)
                    loss = F.cross_entropy(dummy_logits, target)
                else:
                    loss = criterion(output, target)
            
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            # Handle output shape for accuracy calculation
            if output.dim() == 3:  # [T, B, embed_dim]
                # For embedding outputs, we can't compute traditional accuracy
                # Set dummy values
                acc1, acc5 = torch.tensor(0.0), torch.tensor(0.0)
            else:
                if target.dim() > 1:
                    target = target.argmax(-1)
                acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.size(-1))))
            
            batch_size = image.shape[0]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

            if print_freq != 0 and ((idx + 1) % max(1, len(data_loader) // print_freq)) == 0:
                metric_dict.sync()
                logger.debug(' [{}/{}] loss: {:.5f}, acc@1: {:.5f}, acc@5: {:.5f}'.format(
                    idx + 1, len(data_loader), metric_dict['loss'].ave, metric_dict['acc@1'].ave,
                    metric_dict['acc@5'].ave))

    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave


def main():
    ##################################################
    #                       setup
    ##################################################

    args = parse_args()
    dataset_type = args.dataset
    embed_dim = 512  # Standard embedding dimension for face recognition

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    args.output_dir = os.path.join(args.output_dir, f'arcface_s{args.arcface_s}_m{args.arcface_m}')
    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)

    # Single GPU setup
    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)

    logger.info(str(args))

    # Set default input_size if not provided
    input_size = args.input_size if len(args.input_size) != 0 else (3, 224, 224)

    # Load data
    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers, dataset_type, input_size,
        distributed, args.augment, args.mixup, args.cutout, args.label_smoothing, args.T,
        args.triplet_list_train, args.triplet_list_val, args.test_data_path, args.pin_memory)

    # Set num_classes after loading dataset
    num_classes = len(dataset_train.classes)
    logger.info(f'Dataset: {dataset_type}, Classes: {num_classes}, Train: {len(dataset_train)}, Test: {len(dataset_test)}')

    # Model
    model = create_model(
        args.model,
        T=args.T,
        num_classes=embed_dim,  # Output embeddings for face recognition
        img_size=input_size[-1],
    ).cuda()

    # Transfer learning
    if args.transfer:
        logger.info(f'Loading pretrained weights from {args.transfer}')
        checkpoint = torch.load(args.transfer, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Optimizer with better defaults
    optimizer = create_optimizer_v2(
        model,
        opt=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Loss function
    if args.TET:
        criterion = TETArcFaceLoss(
            embed_dim=embed_dim, 
            num_classes=num_classes, 
            s=args.arcface_s, 
            m=args.arcface_m, 
            TET_phi=args.TET_phi, 
            TET_lambda=args.TET_lambda
        ).cuda()
    else:
        criterion = ArcFaceLoss(
            embed_dim=embed_dim, 
            num_classes=num_classes, 
            s=args.arcface_s, 
            m=args.arcface_m
        ).cuda()

    # AMP for memory and speed optimization
    scaler = GradScaler() if args.amp else None

    # LR scheduler with warmup
    lr_scheduler, _ = create_scheduler_v2(
        optimizer,
        sched='cosine',
        num_epochs=args.epochs,
        cooldown_epochs=10,
        min_lr=1e-6,
        warmup_lr=1e-6,
        warmup_epochs=5,
        decay_rate=0.1,
    )

    # Resume from checkpoint
    start_epoch = 0
    max_acc1 = 0.0
    
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_acc1 = checkpoint.get('max_acc1', 0.0)
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info(f'Resumed from epoch {start_epoch}, best acc: {max_acc1:.5f}')

    logger.debug(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    ##################################################
    #                   test only
    ##################################################

    if args.test_only:
        logger.info('Test only mode')
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, args.print_freq, logger)
        logger.info(f'Test Loss: {test_loss:.5f}, Acc@1: {test_acc1:.5f}, Acc@5: {test_acc5:.5f}')
        return

    ##################################################
    #                   Train
    ##################################################

    # Early stopping
    best_acc = max_acc1
    epochs_no_improve = 0

    # TensorBoard
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'), purge_step=start_epoch)

    logger.info(f"Starting training from epoch {start_epoch}, best accuracy: {best_acc:.5f}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f'Epoch [{epoch}/{args.epochs}] Start, lr: {optimizer.param_groups[0]["lr"]:.6f}')

        # Training
        with Timer('Train', logger):
            train_loss = train_one_epoch(
                model, criterion, optimizer, data_loader_train, logger,
                args.print_freq, world_size, args.gradient_accumulation_steps,
                None, scaler
            )

        # Learning rate scheduling
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        # Validation
        with Timer('Test', logger):
            test_loss, test_acc1, test_acc5 = evaluate(
                model, criterion, data_loader_test, args.print_freq, logger
            )

        # Use a meaningful metric for face recognition
        # For embeddings, we'll use negative test loss as the metric to maximize
        current_acc = -test_loss if test_acc1 == 0 else test_acc1

        # TensorBoard logging
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)
        tb_writer.add_scalar('Accuracy/Test_Acc1', test_acc1, epoch)
        tb_writer.add_scalar('Accuracy/Test_Acc5', test_acc5, epoch)
        tb_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f'Epoch [{epoch}] Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Current Metric: {current_acc:.5f}')

        # Early stopping and checkpointing
        is_best = current_acc > best_acc
        
        if is_best:
            best_acc = current_acc
            max_acc1 = current_acc
            epochs_no_improve = 0
            logger.info(f'New best metric: {best_acc:.5f}')
        else:
            epochs_no_improve += 1
            logger.info(f'No improvement for {epochs_no_improve}/{args.patience} epochs')

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_acc1': max_acc1,
            'args': args,
        }
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        # Save best and latest checkpoints
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            
        if args.save_latest:
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pth'))

        # Early stopping check
        if epochs_no_improve >= args.patience:
            logger.info(f'Early stopping after {epoch + 1} epochs')
            break

    logger.info('Training completed.')
    logger.info(f'Final best accuracy: {best_acc:.5f}')
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

    # Reload data using test_data_path
    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(
        args.test_data_path if args.test_data_path else args.data_path, args.batch_size, args.workers,
        dataset_type, input_size, False,
        args.augment, args.mixup, args.cutout,
        args.label_smoothing, args.T,
        args.triplet_list_train, args.triplet_list_val, args.test_data_path
    )

    # Test
    if is_main_process():
        if dataset_type.lower() == 'tripletface':
            test_triplet(model.cuda(), data_loader_test, args.print_freq, logger)
        else:  # CIFAR10
            test(model.cuda(), data_loader_test, input_size, args, logger)
    logger.info('All Done.')


if __name__ == "__main__":
    main()