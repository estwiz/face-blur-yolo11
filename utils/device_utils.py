import torch


def get_device() -> str:
    # Check device
    if torch.cuda.is_available():
        return "cuda"
    # Apple silicon chip
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"