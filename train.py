import os
import os.path as osp
import time
from contextlib import suppress
from typing import Optional, Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from utils import unwrap_model, AverageMeter, DefaultArgs
from model.loss import MultiPosConLossMM
from eval import EvaluatorBase

class Trainer:
    def __init__(
        self, 
        model: "torch.nn.Module | torch.nn.parallel.DistributedDataParallel",
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        grad_scaler: Optional[torch.cuda.amp.GradScaler], 
        lr_scheduler: Optional[CosineAnnealingWarmRestarts], 
        args: DefaultArgs,
        evaluator: Optional[EvaluatorBase] = None,
        eval_freq: Optional[int] = None, # times of eval per epoch
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.evaluator = evaluator
        self.eval_freq = eval_freq

        # precompute evaluation steps
        if evaluator:
            self.eval_steps_per_epoch = np.linspace(0, len(dataloader)-1, self.eval_freq+1).astype(int)[1:] if eval_freq else [len(dataloader)-1]
        else:
            self.eval_steps_per_epoch = []

        self.monitor = {
            'loss': AverageMeter(),
            'batch_time': AverageMeter(),
            'data_time': AverageMeter()
        }
    
    def single_epoch(self, epoch):
        print('Epoch', epoch, 'started.')
        self.current_epoch = epoch
        autocast = torch.cuda.amp.autocast if self.args.precision == 'amp' else suppress

        # setup model
        self.model.train()

        # setup loss
        criterion = MultiPosConLossMM(
            temperature=self.args.local_loss, 
            w1=1.0, 
            w2=1.0,
            distr=self.args.distributed
        )
        
        # setup dataloader
        if self.args.distributed and self.dataloader.sampler is not None:
            self.dataloader.sampler.set_epoch(self.current_epoch)

        # training loop
        end_time = time.time()
        with tqdm(self.dataloader, desc=f'Epoch {self.current_epoch}', unit_scale=self.dataloader.batch_size) as pbar:
            for i, batch in enumerate(pbar):
                self.current_step = len(self.dataloader) * self.current_epoch + i
                videos, texts_sentiment_logits  = batch[0], batch[1]
                texts = texts_sentiment_logits['utt_token']
                audio = texts_sentiment_logits['audio_wav']
                target = texts_sentiment_logits['emotion_idx']

                videos = videos.to(device=self.args.device)
                texts = texts.to(device=self.args.device)
                audio = audio.to(device=self.args.device)
                target = target.to(device=self.args.device)

                self.monitor['data_time'].update(time.time() - end_time)

                self.model.train()
                self.optimizer.zero_grad()

                # forward pass
                with autocast():
                    video_features, text_features, audio_features, logit_scale = self.model(videos, texts, audio)
                    loss_dict = criterion({'image_emb': video_features, 
                                           'text_emb': text_features, 
                                           'audio_emb': audio_features, 
                                           'logit_scale': logit_scale, 
                                           'gt': target})
                    loss = loss_dict['loss']

                # backward pass
                if self.grad_scaler:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step(epoch + i/len(self.dataloader))


                self.monitor['loss'].update(loss.item())
                self.monitor['batch_time'].update(time.time() - end_time)
                end_time = time.time()

                pbar.update()
            # -- end for --
        
        # reset time monitor per epoch
        print('loss', self.monitor['loss'].avg())
        self.monitor['loss'].reset()
        self.monitor['data_time'].reset()
        self.monitor['batch_time'].reset()

        if i in self.eval_steps_per_epoch:
            self.evaluator.model = unwrap_model(self.model)
            metrics = self.evaluator.eval()
            for k, v in metrics.items():
                print(k, v)
    

    def save_checkpoint(self, file_name: str) -> None:

        # construct checkpoint
        data = {}
        data['args'] = vars(self.args)
        data['epoch'] = self.current_epoch
        data['model'] = unwrap_model(self.model).state_dict()
        data['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            data['grad_scaler'] = self.grad_scaler.state_dict()
        if self.lr_scheduler:
            data['lr_scheduler'] = self.lr_scheduler.state_dict()

        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        file_path = osp.join(self.args.ckpt_dir, file_name)
        torch.save(data, file_path)

        # tag the latest checkpoint
        latest = osp.join(self.args.ckpt_dir, 'latest.pt')
        torch.save(data, latest)