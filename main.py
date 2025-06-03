import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
torch.cuda.init()  # Explicitly initialize CUDA context once

import os
import time
import yaml
import random
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
np.int = int
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
from utils.tripletloss import TripletLoss, OnlineTripletLoss
from utils.face_alignment import align_and_crop

# New classes for online hard mining
class LabeledFaceDataset(torch.utils.data.Dataset):
    def __init__(self, triplet_list_file, transform=None, use_face_alignment=True, target_size=(224, 224)):
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.use_face_alignment = use_face_alignment
        self.target_size = target_size
        
        # Collect unique image paths from triplet list
        self.image_paths = set()
        with open(triplet_list_file, 'r') as f:
            for line in f:
                a, p, n = line.strip().split()
                self.image_paths.add(a)
                self.image_paths.add(p)
                self.image_paths.add(n)
        self.image_paths = list(self.image_paths)
        
        # Extract labels from paths (assuming /dataset/label/image.jpg)
        self.labels = [self.extract_label(path) for path in self.image_paths]
        
        # Map labels to indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())
    
    def extract_label(self, path):
        parts = path.split(os.sep)
        if len(parts) >= 2:
            return parts[-2]  # Assumes label is second-to-last part
        raise ValueError(f"Cannot extract label from path: {path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        if self.use_face_alignment:
            try:
                img = align_and_crop(path, self.target_size)
            except Exception as e:
                print(f"Face alignment failed for {path}: {e}")
                img = Image.new('RGB', self.target_size, (128, 128, 128))
        else:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"Image loading failed for {path}: {e}")
                img = Image.new('RGB', self.target_size, (128, 128, 128))
        img = self.transform(img)
        return img, label

class PKSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, P, K):
        self.dataset = dataset
        self.P = P  # Number of classes per batch
        self.K = K  # Number of images per class
        self.num_batches = len(dataset.unique_labels) // P
    
    def __iter__(self):
        for _ in range(self.num_batches):
            sampled_labels = random.sample(self.dataset.unique_labels, self.P)
            batch_indices = []
            for label in sampled_labels:
                indices = self.dataset.label_to_indices[label]
                if len(indices) < self.K:
                    selected = random.choices(indices, k=self.K)
                else:
                    selected = random.sample(indices, self.K)
                batch_indices.extend(selected)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

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
    # Training options
    parser.add_argument('--seed', default=12450, type=int)
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--model', default='spikingresformer_ti', help='model type')
    parser.add_argument('--augment', type=str, help='data augmentation')
    parser.add_argument('--mixup', type=bool, default=False, help='Mixup')
    parser.add_argument('--cutout', type=bool, default=False, help='Cutout')
    parser.add_argument('--label-smoothing', type=float, default=0, help='Label smoothing')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
    parser.add_argument('--print-freq', default=5, type=int, help='Print frequency')
    parser.add_argument('--data-path', default='./datasets')
    parser.add_argument('--output-dir', default='./logs/temp')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--transfer', type=str, help='transfer from pretrained checkpoint')
    parser.add_argument('--input-size', type=int, nargs='+', default=[3, 112, 112])
    parser.add_argument('--distributed-init-mode', type=str, default='env://', choices=['env://', 'none'])
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
    parser.add_argument('--triplet-list-train', type=str, default='', help='Training triplet list')
    parser.add_argument('--triplet-list-val', type=str, default='', help='Validation triplet list')
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET-phi', type=float, default=1.0)
    parser.add_argument('--TET-lambda', type=float, default=0.0)
    parser.add_argument('--save-latest', action='store_true')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    parser.add_argument('--amp', type=bool, default=True, help='Use AMP training')
    parser.add_argument('--sync-bn', action='store_true', help='Use SyncBN training')
    parser.add_argument('--use-online-mining', action='store_true', help='Use online hard mining')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            # Ensure --distributed-init-mode is not overridden by config
            cfg.pop('distributed-init-mode', None)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args

def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s', datefmt=r'%Y-%m-%d %H:%M:%S')
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
    if distributed_init_mode == 'none':
        logger.info('Distributed mode disabled')
        return False, 0, 1, 0
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
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode, world_size=world_size, rank=rank)
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return True, rank, world_size, local_rank

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def custom_collate(batch):
    anchors, positives, negatives = zip(*batch)
    anchor_shapes = [a.shape for a in anchors]
    positive_shapes = [p.shape for p in positives]
    negative_shapes = [n.shape for n in negatives]
    print(f"Batch shapes - Anchors: {anchor_shapes}, Positives: {positive_shapes}, Negatives: {negative_shapes}")
    try:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    except RuntimeError as e:
        print(f"Collate error: {e}")
        raise

def load_data(
    dataset_dir: str,
    batch_size: int,
    workers: int,
    num_classes: int,
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
    use_online_mining: bool = False,
):
    if dataset_type.lower() == 'tripletface':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if use_online_mining:
            train_ds = LabeledFaceDataset(triplet_list_train, transform=transform, use_face_alignment=True, target_size=input_size[-2:])
            P, K = 8, 4  # Adjust P and K based on batch_size (e.g., 32 = 8*4)
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            else:
                train_sampler = PKSampler(train_ds, P, K)
            data_loader_train = torch.utils.data.DataLoader(
                train_ds,
                batch_size=P*K if not distributed else batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler,
            )
        else:
            train_ds = TripletFaceDataset(triplet_list_train, transform=transform, use_face_alignment=True, target_size=input_size[-2:])
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            else:
                train_sampler = None
            data_loader_train = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                num_workers=workers,
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler,
            )
        
        # Validation remains triplet-based
        val_ds = TripletFaceDataset(triplet_list_val, transform=transform, use_face_alignment=True, target_size=input_size[-2:])
        if distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        else:
            test_sampler = None
        data_loader_test = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            sampler=test_sampler,
        )
        dataset_train, dataset_test = train_ds, val_ds
    else:
        raise ValueError(dataset_type)
    return dataset_train, dataset_test, data_loader_train, data_loader_test

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
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            images = torch.cat([anchor, positive, negative], dim=0)
            if scaler is not None:
                with autocast():
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    negative_emb = model(negative)
                    embeddings = torch.cat([anchor_emb, positive_emb, negative_emb], dim=1)
                    loss = criterion(embeddings)
            else:
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                embeddings = torch.cat([anchor_emb, positive_emb, negative_emb], dim=1)
                loss = criterion(embeddings)
            if idx == 0:
                print("Embeddings mean:", embeddings.mean().item())
                print("Embeddings std:", embeddings.std().item())
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
            if idx % 10 == 0:
                torch.cuda.empty_cache()
        batch_size = anchor.size(0)
        if print_freq and ((idx + 1) % int(len(data_loader_train) / print_freq) == 0):
            metric_dict.sync()
            logger.debug(f' [{idx+1}/{len(data_loader_train)}] it/s: {(idx+1)*batch_size*factor/timer_container[0]:.5f}, loss: {metric_dict["loss"].ave:.5f}')
    metric_dict.sync()
    return metric_dict['loss'].ave

def train_one_epoch_online(
    model, criterion, optimizer,
    data_loader_train, logger,
    print_freq, factor,
    scheduler_per_iter=None, scaler=None,
):
    model.train()
    metric_dict = RecordDict({'loss': None})
    timer_container = [0.0]
    model.zero_grad()

    for idx, (images, labels) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            images = images.cuda()
            labels = labels.cuda()
            if scaler is not None:
                with autocast():
                    embeddings = model(images)
                    loss = criterion(embeddings, labels)
            else:
                embeddings = model(images)
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
            if idx % 10 == 0:
                torch.cuda.empty_cache()
        batch_size = images.size(0)
        if print_freq and ((idx + 1) % int(len(data_loader_train) / print_freq) == 0):
            metric_dict.sync()
            logger.debug(f' [{idx+1}/{len(data_loader_train)}] it/s: {(idx+1)*batch_size*factor/timer_container[0]:.5f}, loss: {metric_dict["loss"].ave:.5f}')
    metric_dict.sync()
    return metric_dict['loss'].ave

def evaluate_triplet(model, data_loader, print_freq, logger, criterion_eval):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'accuracy': None})
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(data_loader):
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            images = torch.cat([anchor, positive, negative], dim=0)
            embeddings = model(images)
            loss = criterion_eval(embeddings)
            metric_dict['loss'].update(loss.item())
            T, total, embed_dim = embeddings.shape
            B = total // 3
            embeddings = embeddings.reshape(T, B, 3, embed_dim)
            anchor_e = embeddings[:, :, 0]
            positive_e = embeddings[:, :, 1]
            negative_e = embeddings[:, :, 2]
            pos_dist = torch.sum((anchor_e - positive_e) ** 2, dim=-1)
            neg_dist = torch.sum((anchor_e - negative_e) ** 2, dim=-1)
            correct = (pos_dist < neg_dist).float().mean()
            metric_dict['accuracy'].update(correct.item(), B)
            functional.reset_net(model)
            if print_freq and ((idx + 1) % int(len(data_loader) / print_freq) == 0):
                metric_dict.sync()
                logger.debug(f' [{idx+1}/{len(data_loader)}] loss: {metric_dict["loss"].ave:.5f}, accuracy: {metric_dict["accuracy"].ave:.5f}')
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['accuracy'].ave

def test_triplet(
    model: nn.Module,
    data_loader_test: torch.utils.data.DataLoader,
    print_freq: int,
    logger: logging.Logger,
    criterion_eval,
    margin: float = 0.2,
):
    model.eval()
    metric_dict = RecordDict({'loss': None, 'accuracy': None})
    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(data_loader_test):
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            images = torch.cat([anchor, positive, negative], dim=0)
            embeddings = model(images)
            loss = criterion_eval(embeddings)
            metric_dict['loss'].update(loss.item())
            T, total, D = embeddings.shape
            B = total // 3
            embeddings = embeddings.view(T, B, 3, D)
            a_e = embeddings[:, :, 0]
            p_e = embeddings[:, :, 1]
            n_e = embeddings[:, :, 2]
            pos_dist = torch.sum((a_e - p_e) ** 2, dim=-1)
            neg_dist = torch.sum((a_e - n_e) ** 2, dim=-1)
            correct = (pos_dist < neg_dist).float().mean()
            metric_dict['accuracy'].update(correct.item(), B)
            functional.reset_net(model)
            if print_freq and ((idx + 1) % int(len(data_loader_test) / print_freq) == 0):
                metric_dict.sync()
                logger.debug(f' [{idx+1}/{len(data_loader_test)}] loss: {metric_dict["loss"].ave:.5f}, accuracy: {metric_dict["accuracy"].ave:.5f}')
    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['accuracy'].ave

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    safe_makedirs(args.output_dir)
    logger = setup_logger(args.output_dir)
    distributed, rank, world_size, local_rank = init_distributed(logger, args.distributed_init_mode)
    logger.info(str(args))

    dataset_type = args.dataset
    one_hot = None
    if dataset_type.lower() == 'tripletface':
        num_classes = 512
        input_size = (3, 224, 224)
    else:
        raise ValueError(dataset_type)
    if len(args.input_size) != 0:
        input_size = args.input_size

    dataset_train, dataset_test, data_loader_train, data_loader_test = load_data(
        args.data_path, args.batch_size, args.workers, num_classes, dataset_type, input_size,
        distributed, args.augment, args.mixup, args.cutout, args.label_smoothing, args.T,
        args.triplet_list_train, args.triplet_list_val, args.use_online_mining)
    logger.info('dataset_train: {}, dataset_test: {}'.format(len(dataset_train), len(dataset_test)))

    model = create_model(args.model, T=args.T, num_classes=num_classes, img_size=input_size[-1]).cuda()
    if args.transfer:
        checkpoint = torch.load(args.transfer, map_location='cpu')
        model.transfer(checkpoint['model'])
    optimizer = create_optimizer_v2(model, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    margin = 0.2
    criterion = OnlineTripletLoss(margin=margin)
    criterion_eval = OnlineTripletLoss(margin=margin)
    if args.TET:
        criterion = CriterionWarpper(criterion, args.TET, args.TET_phi, args.TET_lambda)
        criterion_eval = CriterionWarpper(criterion_eval)
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    lr_scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.epochs, cooldown_epochs=10, min_lr=1e-5, warmup_lr=1e-5, warmup_epochs=3)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if distributed and not args.test_only:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
        model_without_ddp = model.module
    scheduler_per_iter = None
    scheduler_per_epoch = None
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

    if args.test_only:
        if distributed:
            logger.error('Using distribute mode in test, abort')
            return
        test_triplet(model_without_ddp, data_loader_test, args.print_freq, logger)
        return

    tb_writer = None
    if is_main_process():
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'), purge_step=start_epoch)

    logger.info("[Train]")
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch, optimizer.param_groups[0]["lr"]))
        with Timer(' Train', logger):
            if args.use_online_mining:
                train_loss = train_one_epoch_online(model, criterion, optimizer, data_loader_train, logger, args.print_freq, world_size, scheduler_per_iter, scaler)
            else:
                train_loss = train_one_epoch_triplet(model, criterion, optimizer, data_loader_train, logger, args.print_freq, world_size, scheduler_per_iter, scaler)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if scheduler_per_epoch is not None:
                scheduler_per_epoch.step()
        with Timer(' Test', logger):
            test_loss, test_acc = evaluate_triplet(model, data_loader_test, args.print_freq, logger, criterion_eval)
        if is_main_process() and tb_writer is not None:
            tb_record_triplet(tb_writer, train_loss, test_loss, test_acc, epoch)
        logger.info(' Test loss: {:.5f}, Accuracy: {:.5f}'.format(test_loss, test_acc))
        checkpoint = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'max_acc1': max_acc1}
        if lr_scheduler is not None:
            checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
        if test_acc > max_acc1:
            max_acc1 = test_acc
            best_ckpt = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'max_acc1': max_acc1}
            if lr_scheduler is not None:
                best_ckpt['lr_scheduler'] = lr_scheduler.state_dict()
            save_on_master(best_ckpt, os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'))
        if args.save_latest:
            latest_ckpt = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'max_acc1': max_acc1}
            if lr_scheduler is not None:
                latest_ckpt['lr_scheduler'] = lr_scheduler.state_dict()
            save_on_master(latest_ckpt, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
            logger.info('Training completed.')

    del model, model_without_ddp
    model = create_model(args.model, T=args.T, num_classes=512, img_size=input_size[-1])
    try:
        checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint_max_acc1.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    except:
        logger.warning('Cannot load max acc1 model, skip test.')
        logger.warning('Exit.')
        return
    del dataset_train, dataset_test, data_loader_train, data_loader_test
    _, _, _, data_loader_test = load_data(args.data_path, args.batch_size, args.workers, num_classes, dataset_type, input_size, False,
                                          args.augment, args.mixup, args.cutout, args.label_smoothing, args.T, args.triplet_list_train, args.triplet_list_val)
    if is_main_process():
        test_triplet(model.cuda(), data_loader_test, args.print_freq, logger, criterion_eval)
    logger.info('All Done.')

if __name__ == "__main__":
    main()