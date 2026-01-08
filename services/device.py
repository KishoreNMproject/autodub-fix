import torch

def get_device():
    """
    Auto-detect best available device:
    - NVIDIA CUDA
    - AMD ROCm
    - CPU fallback
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"GPU (CUDA): {name}"

    # ROCm uses the same torch.cuda API
    if torch.version.hip is not None:
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"GPU (ROCm): {name}"

    return torch.device("cpu"), "CPU"
