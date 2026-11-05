import json
import platform
import sys
import torch

print(json.dumps({
    "python": sys.version.split()[0], "platform": platform.platform(),
    "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
