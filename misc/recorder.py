import math
import os

import torch


class Recorder:
    def __init__(self, trainer):
        self.trainer = trainer

        if trainer.ckpt:
            self.loss_min = trainer.ckpt["loss"]
        else:
            self.loss_min = math.inf

    def __call__(self, loss, epoch):
        self.save_dict = {
            "epoch": epoch,
            "loss": self.loss_min,
            "model": self.trainer.model.module.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "scheduler": self.trainer.scheduler.state_dict(),
        }
        if loss < self.loss_min:
            self.save_ckpt_best(loss)
        self.save_ckpt()

    def save_ckpt_best(self, loss):
        print(
            "Loss decreased ({0:.4f} => {1:.4f}). Save as the best checkpoint.".format(
                self.loss_min, loss
            )
        )
        self.loss_min = loss
        self.save_dict["loss"] = loss
        torch.save(
            self.save_dict, os.path.join(self.trainer.result_path, "ckpt_best.pt")
        )

    def save_ckpt(self):
        torch.save(self.save_dict, os.path.join(self.trainer.result_path, "ckpt.pt"))
