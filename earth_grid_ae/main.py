import argparse
import datetime
import os
import random
from time import strftime

import torch
from torch import distributed, multiprocessing
from train import Trainer


def setup(rank, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port

    distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(hours=1),
    )


def cleanup():
    distributed.destroy_process_group()


def main(rank, args):
    setup(rank, args)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    trainer = Trainer(args, rank)
    trainer.train()

    cleanup()


def create_args(accumulate_size, batch_size, epoch, lr, loss):
    args = argparse.ArgumentParser()

    args.add_argument("--port", default=str(random.randint(1024, 65535)), type=str)

    args.add_argument("--resume", default="", type=str)
    args.add_argument("--result_dir", default="result", type=str)
    args.add_argument("--result_name", default="earth_grid_ae", type=str)
    args.add_argument("--result_date", default=strftime("%y%m%d-%H%M%S"), type=str)
    args.add_argument("--seed", default=2024, type=int)

    args.add_argument("--cuda_device", default="", type=str)

    args.add_argument("--accumulate_size", default=accumulate_size, type=int)
    args.add_argument("--batch_size", default=batch_size, type=int)
    args.add_argument("--epoch", default=epoch, type=int)
    args.add_argument("--lr", default=lr, type=float)
    args.add_argument("--loss", default=loss, type=str)

    args.add_argument("--latent_size", default=1024, type=int)

    return args


if __name__ == "__main__":
    accumulate_size = 2
    batch_size = 4
    epoch = 50
    lr = 1e-4
    loss = "l2"

    args = create_args(accumulate_size, batch_size, epoch, lr, loss).parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.world_size = len(args.cuda_device.split(","))

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
