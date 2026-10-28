import importlib

import torch


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        module = importlib.import_module(f"models.{self.args.model}")
        return module.Model(self.args).float()

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
            return device
        print("Use CPU")
        return torch.device("cpu")
