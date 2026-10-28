import hashlib
import importlib.metadata
import os
import platform

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_audit(args):
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(args.root_path, data_path)
    device = torch.cuda.get_device_name(args.gpu) if args.use_gpu else "CPU"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas")
        },
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "dataset_sha256": sha256_file(data_path),
        "dataset_bytes": os.path.getsize(data_path),
    }
