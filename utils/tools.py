import os
import random
import time

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == "none":
        return
    if args.lradj == "type1":
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == "cosine":
        lr = args.learning_rate / 2 * (1 + np.cos(epoch / args.train_epochs * np.pi))
    else:
        raise ValueError(f"Unsupported lradj: {args.lradj}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Updating learning rate to {lr}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
