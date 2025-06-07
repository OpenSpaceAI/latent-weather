import json
import logging
import os
import sys

import torch
from dataset import Dataset
from model import Autoencoder
from munch import DefaultMunch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import get_cosine_schedule_with_warmup

sys.path.append(".")

from misc.metric import ACC, MAE, MSE, RMSE
from misc.recorder import Recorder
from misc.util import check_dir, print_log, set_seed


class Trainer:
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.ckpt = False
        self.device = f"cuda:{self.rank}"
        self.kwargs = {"device": self.device, "dtype": torch.float}

        if self.args.resume:
            with open(
                os.path.join(self.args.result_dir, self.args.resume, "parameter.json"),
                encoding="utf-8",
                mode="r",
            ) as f:
                self.args = DefaultMunch.fromDict(json.load(f))

            self.ckpt = torch.load(
                os.path.join(self.args.result_dir, args.resume, "ckpt.pt"),
                map_location="cpu",
                weights_only=False,
            )

        set_seed(self.args.seed)

        self.result_fullname = self.args.result_date
        if self.args.result_name != "":
            self.result_fullname += "_" + self.args.result_name

        self.result_path = os.path.join(self.args.result_dir, self.result_fullname)

        if self.rank == 0:
            check_dir(self.result_path)

            with open(
                os.path.join(self.result_path, "parameter.json"),
                encoding="utf-8",
                mode="w",
            ) as f:
                json.dump(vars(self.args), f, indent=2)

            logging.basicConfig(
                level=logging.INFO,
                filename=os.path.join(self.result_path, "out.log"),
                filemode="a",
                format="%(asctime)s - %(message)s",
                encoding="utf-8",
            )
            self.recorder = Recorder(self)

        self.get_dataset()
        self.build_model()
        self.select_criterion()

    def get_dataset(self):
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        self.train_data = Dataset("train")
        self.val_data = Dataset("val")

        self.train_sampler = DistributedSampler(
            self.train_data,
            num_replicas=self.args.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.args.seed,
        )
        self.val_sampler = DistributedSampler(
            self.val_data,
            num_replicas=self.args.world_size,
            rank=self.rank,
            shuffle=False,
            seed=self.args.seed,
        )

        num_workers = 4

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=num_workers,
            sampler=self.train_sampler,
            generator=generator,
            persistent_workers=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.args.batch_size,
            num_workers=num_workers,
            sampler=self.val_sampler,
            generator=generator,
            persistent_workers=True,
            pin_memory=True,
        )

    def build_model(self):
        self.shape = (109, 36, 72)

        self.model = Autoencoder(self.args.latent_size, self.kwargs).to(**self.kwargs)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.ckpt:
            self.model.load_state_dict(self.ckpt["model"])

        self.model = DistributedDataParallel(
            self.model, device_ids=[self.rank], output_device=self.rank
        )

        total_step = (
            len(self.train_loader) // self.args.accumulate_size * self.args.epoch
        )
        if len(self.train_loader) % self.args.accumulate_size != 0:
            total_step += self.args.epoch

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_step // 10,
            num_training_steps=total_step,
        )

        if self.ckpt:
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])

    def select_criterion(self):
        self.rmse = RMSE("mars_grid", self.device)
        self.acc = ACC("mars_grid", self.device)
        self.mean = torch.load(
            "/path/to/dataset/mars_grid/ref/mean.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.std = torch.load(
            "/path/to/dataset/mars_grid/ref/std.pt",
            map_location=self.device,
            weights_only=True,
        )

        if self.args.loss == "l1":
            self.criterion = MAE
        elif self.args.loss == "l2":
            self.criterion = MSE
        else:
            exit(1)

    def train(self):
        if self.ckpt:
            epoch_start = self.ckpt["epoch"] + 1
        else:
            epoch_start = 0

        if self.rank == 0:
            range_bar = trange(
                epoch_start,
                self.args.epoch,
                bar_format="{l_bar}{bar}{r_bar} ETA: {eta:%y/%m/%d %H:%M}",
            )
            range_bar.set_description("Train")
        else:
            range_bar = range(epoch_start, self.args.epoch)

        for epoch in range_bar:
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            self.optimizer.zero_grad()

            loss = 0
            cnt = 0

            if self.rank == 0:
                dataloader = tqdm(self.train_loader)
            else:
                dataloader = self.train_loader

            for i, data in enumerate(dataloader):
                cnt += 1

                input_raw = data["input_raw"].to(**self.kwargs)

                predict_raw = self.model(input_raw)

                loss_now = self.criterion(predict_raw, input_raw)

                loss_now = loss_now / self.args.accumulate_size

                loss_now.backward()

                loss_real = loss_now * self.args.accumulate_size
                loss += loss_real

                if (i + 1) % self.args.accumulate_size == 0 or (i + 1) == len(
                    dataloader
                ):
                    clip_grad_norm_(
                        self.model.parameters(), max_norm=1, error_if_nonfinite=True
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.rank == 0:
                    dataloader.set_description(
                        f"Epoch: {epoch + 1} | Train loss: {loss_real:.4f}"
                    )

            loss /= cnt

            loss_val = self.val(epoch)

            if self.rank == 0:
                print_log(
                    f"Epoch: {epoch + 1} | Train loss: {loss:.4f}, Val loss: {loss_val:.4f}"
                )
                self.recorder(loss_val, epoch)

        distributed.barrier()
        self.test()
        self.save_latent()

    def val(self, epoch):
        with torch.no_grad():
            self.model.eval()

            loss = 0
            cnt = 0

            if self.rank == 0:
                dataloader = tqdm(self.val_loader)
            else:
                dataloader = self.val_loader

            for i, data in enumerate(dataloader):
                cnt += 1

                input_raw = data["input_raw"].to(**self.kwargs)

                predict_raw = self.model(input_raw)

                predict_raw = predict_raw * (self.std + 1e-8) + self.mean
                input_raw = input_raw * (self.std + 1e-8) + self.mean

                loss_now = 0
                for j in range(self.shape[0]):
                    loss_now = loss_now - self.acc(
                        predict_raw[:, j], input_raw[:, j], self.mean[j]
                    )

                if self.rank == 0:
                    dataloader.set_description(
                        f"Epoch: {epoch + 1} | Val loss: {loss_now:.4f}"
                    )

                loss += loss_now

        loss /= cnt

        return loss

    def test(self):
        model = torch.load(
            os.path.join(self.result_path, "ckpt_best.pt"),
            map_location="cpu",
            weights_only=False,
        )["model"]
        self.model.module.load_state_dict(model)

        with torch.no_grad():
            self.model.eval()

            cnt = torch.zeros(self.shape[0], dtype=torch.int, device=self.device)
            mae = torch.zeros(self.shape[0], dtype=torch.float, device=self.device)
            rmse = torch.zeros(self.shape[0], dtype=torch.float, device=self.device)
            acc = torch.zeros(self.shape[0], dtype=torch.float, device=self.device)

            if self.rank == 0:
                dataloader = tqdm(self.val_loader)
                dataloader.set_description("Test")
            else:
                dataloader = self.val_loader

            for i, data in enumerate(dataloader):
                input_raw = data["input_raw"].to(**self.kwargs)

                predict_raw = self.model(input_raw)

                predict_raw = predict_raw * (self.std + 1e-8) + self.mean
                input_raw = input_raw * (self.std + 1e-8) + self.mean

                for j in range(input_raw.shape[0]):
                    for k in range(self.shape[0]):
                        cnt[k] += 1

                        mae_now = MAE(predict_raw[j][k], input_raw[j][k])
                        mae[k] += mae_now

                        rmse_now = self.rmse(predict_raw[j][k], input_raw[j][k])
                        rmse[k] += rmse_now

                        acc_now = self.acc(
                            predict_raw[j][k], input_raw[j][k], self.mean[k]
                        )
                        acc[k] += acc_now

        distributed.barrier()

        if self.rank == 0:
            cnt_list = [
                torch.zeros(cnt.shape, dtype=torch.int, device=self.device)
                for _ in range(self.args.world_size)
            ]
            mae_list = [
                torch.zeros(mae.shape, dtype=torch.float, device=self.device)
                for _ in range(self.args.world_size)
            ]
            rmse_list = [
                torch.zeros(rmse.shape, dtype=torch.float, device=self.device)
                for _ in range(self.args.world_size)
            ]
            acc_list = [
                torch.zeros(acc.shape, dtype=torch.float, device=self.device)
                for _ in range(self.args.world_size)
            ]

            distributed.gather(cnt, cnt_list, dst=0)
            distributed.gather(mae, mae_list, dst=0)
            distributed.gather(rmse, rmse_list, dst=0)
            distributed.gather(acc, acc_list, dst=0)
        else:
            distributed.gather(cnt, dst=0)
            distributed.gather(mae, dst=0)
            distributed.gather(rmse, dst=0)
            distributed.gather(acc, dst=0)

        if self.rank == 0:
            cnt = torch.zeros(cnt.shape, dtype=torch.int, device=self.device)
            mae = torch.zeros(mae.shape, dtype=torch.float, device=self.device)
            rmse = torch.zeros(rmse.shape, dtype=torch.float, device=self.device)
            acc = torch.zeros(acc.shape, dtype=torch.float, device=self.device)

            for i in range(self.shape[0]):
                for j in range(self.args.world_size):
                    cnt[i] += cnt_list[j][i]

                    mae[i] += mae_list[j][i]
                    rmse[i] += rmse_list[j][i]
                    acc[i] += acc_list[j][i]

            for i in range(self.shape[0]):
                mae[i] /= cnt[i]
                rmse[i] /= cnt[i]
                acc[i] /= cnt[i]

            for i in range(self.shape[0]):
                logging.info(
                    f"Test channel {i + 1}: num {cnt[i]}, MAE {mae[i]:.4f}, RMSE {rmse[i]:.4f}, ACC {acc[i]:.4f}"
                )

            save_dict = {"cnt": cnt, "mae": mae, "rmse": rmse, "acc": acc}
            torch.save(save_dict, os.path.join(self.result_path, "test.pt"))

    def save_latent(self):
        if self.rank == 0:
            with torch.no_grad():
                self.model.eval()

                data_dir = "/zssd/dataset/mars/tensor_fp16"

                data_list = os.listdir(data_dir)
                data_list.sort()

                dataloader = tqdm(data_list)
                dataloader.set_description("Save")

                data = []

                for i in dataloader:
                    input_raw = torch.load(
                        os.path.join(data_dir, i),
                        map_location=self.device,
                        weights_only=True,
                    )
                    predict_raw = self.model.module.encoder(
                        input_raw.to(dtype=torch.float).unsqueeze(0)
                    )
                    data.append(predict_raw)

                data = torch.cat(data, dim=0)

                torch.save(data, os.path.join(self.result_path, "latent.pt"))
