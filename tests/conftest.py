import sys
from types import ModuleType

# Provide a lightweight torch stub when PyTorch is not loadable.
# The stub exposes a real Tensor class because scipy's array_api_compat
# uses issubclass() against torch.Tensor at import time.
try:
    import torch as _real_torch  # noqa: F401
except (ImportError, OSError):
    _torch = ModuleType("torch")
    _torch.Tensor = type("Tensor", (), {})
    _torch.nn = ModuleType("torch.nn")
    _torch.nn.Module = type("Module", (), {})
    _torch.load = lambda *a, **kw: None
    _torch.save = lambda *a, **kw: None
    _torch.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
    _torch.float32 = "float32"

    import numpy as _np

    def _fake_tensor(data, **kw):
        return _np.asarray(data)

    _torch.tensor = _fake_tensor
    sys.modules["torch"] = _torch
    sys.modules["torch.nn"] = _torch.nn
