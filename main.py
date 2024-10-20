import argparse
import os
import yaml
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model.clip import EmotionCLIP
from model.adapter import *
from data.meld_data import MELD, EMOTION_CLASS_NAMES
from train import Trainer
from eval import LinearProbClassifier
from utils import DefaultArgs, set_random_seed
from distr import is_master, init_distributed_device, world_info_from_env, setup_print_for_distributed

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():

    args = DefaultArgs()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='train or eval.')
    cargs = parser.parse_args()
    for k, v in vars(cargs).items():
        setattr(args, k, v)

    # setup distributed
    #init_distributed_device(args)
    #setup_print_for_distributed(args)

    # setup model
    model = EmotionCLIP(
        backbone_config=args.backbone_config,
        backbone_checkpoint=args.backbone_checkpoint,
        temporal_fusion=args.temporal_fusion,
        video_len=args.video_len,
        head_nlayer=args.head_nlayer,
        reset_logit_scale=args.reset_logit_scale,
    ).to(args.device)

    if args.pre_trained:
        print('load pretrained checkpoints')
        ckpt_emotionclip= torch.load(args.pre_trained, map_location='cpu')
        model.load_state_dict(ckpt_emotionclip['model'], strict=False)
    
    # setup data
    train_dataset = MELD(
        data_dir=args.data_dir,
        split='train',
        sampling_strategy=args.sampling_strategy,
        dense_sampling_interval=args.dense_sampling_interval,
        video_len=args.video_len,
        target=args.target,
        audio_sample_rate=args.audio_sample_rate,
    )
    
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.train_loader_workers,
            pin_memory=args.pin_memory
        )

    if args.mode == "train" or args.mode == "eval":

        finetune_dataset = MELD(
            data_dir=args.data_dir,
            split='test',
            sampling_strategy=args.sampling_strategy,
            dense_sampling_interval=args.dense_sampling_interval,
            video_len=args.video_len,
            target='emotion_idx',
            audio_sample_rate=args.audio_sample_rate,
        )

        val_dataset = MELD(
            data_dir=args.data_dir,
            split='dev',
            sampling_strategy=args.sampling_strategy,
            dense_sampling_interval=args.dense_sampling_interval,
            video_len=args.video_len,
            target='emotion_idx',
            audio_sample_rate=args.audio_sample_rate,
        )

        finetune_dataloader = DataLoader(
                dataset=finetune_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=False,
                num_workers=args.val_loader_workers,
                pin_memory=args.pin_memory
            )
        
        val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=False,
                num_workers=args.val_loader_workers,
                pin_memory=args.pin_memory
            )
        
        linear_clf = LinearProbClassifier(
                model=model,
                dataloaders=[finetune_dataloader, val_dataloader],
                args=args
            )

    if args.mode == "eval":
        metrics = linear_clf.eval()
        for k, v in metrics.items():
            print(k, v)
        
    if args.mode == "adapter":
        
        finetune_dataset = MELD(
            data_dir=args.data_dir,
            split='test',
            sampling_strategy=args.sampling_strategy,
            dense_sampling_interval=args.dense_sampling_interval,
            video_len=args.video_len,
            target=args.target,
            audio_sample_rate=args.audio_sample_rate,
        )

        val_dataset = MELD(
            data_dir=args.data_dir,
            split='dev',
            sampling_strategy=args.sampling_strategy,
            dense_sampling_interval=args.dense_sampling_interval,
            video_len=args.video_len,
            target=args.target,
            audio_sample_rate=args.audio_sample_rate,
        )

        finetune_dataloader = DataLoader(
                dataset=finetune_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=False,
                num_workers=args.val_loader_workers,
                pin_memory=args.pin_memory
            )
        
        val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=False,
                drop_last=False,
                num_workers=args.val_loader_workers,
                pin_memory=args.pin_memory
            )
        
        # Textual features
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(EMOTION_CLASS_NAMES, model, args.device)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")

        adapter_cfg = yaml.load(open(args.adpter_config, 'r'), Loader=yaml.Loader)
        cache_dir = os.path.join('/mnt/ff1f01b3-85e2-407c-8f5d-cdcee532daa5/emodet_cache', 'meld_kv')
        os.makedirs(cache_dir, exist_ok=True)
        adapter_cfg['cache_dir'] = cache_dir

        cache_keys, cache_values = build_cache_model(adapter_cfg, model, train_dataloader, args.device)

        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        val_features, val_labels = pre_load_features(adapter_cfg, "val", model, val_dataloader, args.device)
        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        test_features, test_labels = pre_load_features(adapter_cfg, "test", model, finetune_dataloader, args.device)

        # ------------------------------------------ Tip-Adapter ------------------------------------------
        run_tip_adapter(adapter_cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
        run_tip_adapter_F(adapter_cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, model, train_dataloader, args.device)


    if args.mode == "train":

        if args.distributed:
            model = DistributedDataParallel(
                module=model,
                device_ids=[args.device] if args.device_mode=='cuda' else None,
                static_graph=args.ddp_static_graph
            )

        ## collect the trainable params
        # helper filters
        is_frame_params = lambda n, p: 'backbone.visual' in n
        is_text_params = lambda n, p: 'backbone' in n and 'visual' not in n and 'logit_scale' not in n
        is_audio_params = lambda n, p: 'audio_model' in n
        is_temporal_params = lambda n, p: 'visual_head' in n and 'logit_scale' not in n
        is_gain_or_bias_params = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

        # paramter groups
        all_named_parameters = list(model.named_parameters())

        # freeze the CLIP backbone
        for name, layer in model.named_modules():
            for params in layer.parameters():
                params.requires_grad = False

        target_module_names = ['audio_model', 'visual_head', 'backbone.transformer.resblocks.11']
        for name, layer in model.named_modules():
            if any([m in name for m in target_module_names]):
                for params in layer.parameters():
                    params.requires_grad = True
            

        #frame_gb_params = [
        #    p for n, p in all_named_parameters if is_frame_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
        #]
        #frame_rest_params = [
        #    p for n, p in all_named_parameters if is_frame_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
        #]
        text_gb_params = [
            p for n, p in all_named_parameters if is_text_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        text_rest_params = [
            p for n, p in all_named_parameters if is_text_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        audio_gb_params = [
            p for n, p in all_named_parameters if is_audio_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        audio_rest_params = [
            p for n, p in all_named_parameters if is_audio_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        temporal_gb_params = [
            p for n, p in all_named_parameters if is_temporal_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        temporal_rest_params = [
            p for n, p in all_named_parameters if is_temporal_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
        ]
        logit_scale_params = [
            p for n, p in all_named_parameters if 'logit_scale' in n and p.requires_grad
        ]
        # setup optimizer
        param_groups_for_optimizer = [
            #{'params': frame_gb_params, 'lr': args.lr_backbone_gb, 'weight_decay': 0.},
            #{'params': frame_rest_params, 'lr': args.lr_backbone_rest, 'weight_decay': args.weight_decay_backbone},

            {'params': text_gb_params, 'lr': args.lr_backbone_gb, 'weight_decay': 0.},
            {'params': text_rest_params, 'lr': args.lr_backbone_rest, 'weight_decay': args.weight_decay_backbone},

            {'params': audio_gb_params, 'lr': args.lr_head_gb, 'weight_decay': 0.},
            {'params': audio_rest_params, 'lr': args.lr_head_rest, 'weight_decay': args.weight_decay_backbone},

            {'params': temporal_gb_params, 'lr': args.lr_head_gb, 'weight_decay': 0.},
            {'params': temporal_rest_params, 'lr': args.lr_head_rest, 'weight_decay': args.weight_decay_head},

            {'params': logit_scale_params, 'lr': args.lr_backbone_rest, 'weight_decay': 0.}
        ]

        optimizer = torch.optim.AdamW(
            param_groups_for_optimizer,
            betas = (args.adamw_beta1, args.adamw_beta2),
            eps = args.adamw_eps
        )
        # setup grad_scaler for AMP
        grad_scaler = torch.cuda.amp.GradScaler() if args.precision == 'amp' else None
        # setup lr_scheduler
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.max_epochs,
            T_mult=1,
            eta_min=args.lr_min,
        )

        # create trainer
        trainer = Trainer(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            args=args,
            evaluator=linear_clf if args.enable_eval else None,
            eval_freq=args.eval_freq
        )

        # main loop
        for current_epoch in range(args.start_epoch, args.max_epochs):
            trainer.single_epoch(current_epoch)
            if args.ckpt_dir:
                trainer.save_checkpoint(file_name=f'epoch{current_epoch}.pt')

if __name__ == '__main__':
    main()