import argparse
import datetime
import json
# import random
import time
from pathlib import Path
import numpy as np

# import datasets
import util.misc as utils
from tools.config import config2args, setargs2config, saveconfig
from models import build_model
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, ddp_train_one_epoch
from tools.loadargs import *

#
#from torch.utils.data import DataLoader, DistributedSampler
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# import torch.nn as nn
# import torch.optim as optim


def main(args, config):
    print("Detr_version : ",args.detr_version)
    print(args)
    Train_start_time = datetime.datetime.now() 

    # Build dataset
    dataset_train = build_dataset(image_set='train', args=args)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    
    checkpoint_dir = args.checkpoint_dir+args.testname + "/"

    if args.dist == "OFF":
        device = torch.device(args.device)
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        model_without_ddp = model

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\033[93m'+'number of params: {}'.format(n_parameters) +'\033[93m')

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        

        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        if args.resume == "ON":
            print('\033[96m'+" >>> Resuming Exsisting model Training "+'\033[0m')
            filename = args.testname +'.pth'
            paused_checkpoint_path = checkpoint_dir + filename
            checkpoint = torch.load(paused_checkpoint_path)
            state_dict = checkpoint['model']

            model_without_ddp.load_state_dict(state_dict)
            
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        elif args.resume == "OFF":
            print('\033[92m'+" >>> Started New model Training"+'\033[0m')


        print('\033[94m' + ">>> Start training . . . ."+'\033[0m')
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            print("epoch : ",epoch)
            train_stats = ddp_train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            lr_scheduler.step()
            
            # if args.checkpoint_dir:
            filename = args.testname +'.pth'
            checkpoint_path = checkpoint_dir + filename

            if (epoch + 1) % 2 == 0:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
            #             'epoch': epoch,
            #             'n_parameters': n_parameters}

            # if checkpoint_dir and utils.is_main_process():
            #     with (Path(checkpoint_dir) / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")
            saveconfig(args, config, Train_start_time, epoch)

    if args.dist == "ON":
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        #utils.init_distributed_mode(args)
    #print("git:\n  {}\n".format(utils.get_sha()))

    # if args.frozen_weights is not None:
    #     assert args.masks, "Frozen training is meant for segmentation only"

        gpu_id = int(os.environ["LOCAL_RANK"])
        print("Gpu_id ============  ", gpu_id)
        args.device = gpu_id
        device = torch.device(args.device)
        model, criterion, postprocessors = build_model(args)
        model.to(gpu_id)
        model = DDP(model, device_ids=[gpu_id]) #ddp wrapper
        model_with_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\033[93m'+'number of params: {}'.format(n_parameters) +'\033[93m')

        param_dicts = [
            {"params": [p for n, p in model_with_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_with_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=utils.collate_fn, pin_memory=True,shuffle=False,sampler=DistributedSampler(dataset_train))

        if args.resume == "ON":
            print('\033[96m'+" >>> Resuming Exsisting ddp model Training "+'\033[0m')
            filename = args.testname +'.pth'
            paused_checkpoint_path = checkpoint_dir + filename
            checkpoint = torch.load(paused_checkpoint_path)
            state_dict = checkpoint['model']

            model_with_ddp.load_state_dict(state_dict)
            
            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if args.resume == "OFF":
            print('\033[92m'+" >>> Started New ddp model Training "+'\033[0m')

        print('\033[94m' + ">>> Start training . . . ."+'\033[0m')
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            #     sampler_train.set_epoch(epoch)
            data_loader_train.sampler.set_epoch(epoch)

            train_stats = ddp_train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
            lr_scheduler.step()
            
            # if args.checkpoint_dir:
            filename = args.testname +'.pth'
            checkpoint_path = checkpoint_dir + filename

            if gpu_id == 0 and (epoch + 1) % 2 == 0:
                torch.save({
                    'model': model_with_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
            #             'epoch': epoch,
            #             'n_parameters': n_parameters}

            # if checkpoint_dir and utils.is_main_process():
            #     with (Path(checkpoint_dir) / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")
            # saveconfig(args, config, Train_start_time, epoch)


        destroy_process_group()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    config2args(args)
    config = setargs2config(args)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
