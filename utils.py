from __future__ import annotations
import random
from typing import Callable, Optional, Any, List, Tuple

import argparse
import os
import os.path as osp

import numpy as np
import torch

class DefaultArgs(argparse.Namespace):

    # general setting
    PROJECT_PATH: str = '.'
    device_mode: str = 'cuda'
    device: Optional[str] = 'cuda:0'
    precision: str = 'amp'
    seed: Optional[int] = 2022
    deterministic: bool = False

    # distributed
    distributed: Optional[bool] = False
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    no_set_device_rank: bool = False
    #use_bn_sync: bool = True
    ddp_static_graph: bool = True
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None

    # dataset
    data_dir: str = '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/MELD.Raw'
    split: str = 'train'
    sampling_strategy: str = 'uniform'
    dense_sampling_interval: int = 4
    video_len: int = 8
    target: str = 'multimodal_finetune'
    audio_sample_rate: int = 16000
    

    # dataloader
    batch_size: int = 256
    train_loader_workers: int = 8
    val_loader_workers: int = 8
    pin_memory: bool = True

    # model
    pre_trained: Optional[str] = None #'/home/minxiao/workspace/emo_det/EmotionCLIP/checkpoints_download/emotionclip_latest.pt'
    backbone_config: str = osp.join(PROJECT_PATH, 'model/model_configs/ViT-B-32.json')
    backbone_checkpoint: str = osp.join(PROJECT_PATH, 'checkpoint/vit_b_32-laion2b_e16-af8dbd0c.pth')
    temporal_fusion: str = 'transformer'
    head_nlayer: int = 6
    ckpt_dir: str = '/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache/checkpoints'

    # loss
    local_loss: bool = False
    gather_with_grad: bool = False
    loss_reweight_scale: Optional[float] = 1.2

    # training
    start_epoch: int = 0
    max_epochs: int = 50
    lr_backbone_gb: float = 5e-5
    lr_backbone_rest: float = 1e-8
    lr_head_gb: float = 5e-4
    lr_head_rest: float = 1e-7
    lr_min: float = 1e-10
    weight_decay_backbone: float = 0.1
    weight_decay_head: float = 0.1
    adamw_beta1: float = 0.98
    adamw_beta2: float = 0.9
    adamw_eps: float = 1e-6
    reset_logit_scale: bool = True

    # evaluation
    enable_eval: bool = True
    eval_freq: int = 50

    # adapter
    adpter_config: str = '/home/minxiao/workspace/EmoAsst/model/model_configs/meld_config_tip.yaml'

class AverageMeter:
    def __init__(self):
        self._buffer = []

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, val: float) -> None:
        self._buffer.append(val)

    @property
    def val(self) -> float:
        return self._buffer[-1]

    def avg(self, window_size:Optional[int] = None) -> float:
        if window_size is None:
            return np.mean(self._buffer)
        else:
            return np.mean(self._buffer[-window_size:])

def unwrap_model(model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel) -> torch.nn.Module:
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def set_random_seed(seed: int = 2022, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
