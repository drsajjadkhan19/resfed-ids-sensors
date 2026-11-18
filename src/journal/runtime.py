"""Run-time settings shared by every study, so that results do not depend on the machine.

* ``workers()`` sets the process pool size from ``RESFED_WORKERS`` (default: all
  cores). Every job seeds itself, so results do not depend on the pool size.
* ``deterministic()`` pins PyTorch to one thread and deterministic kernels on the
  CPU. Accelerators (MPS, CUDA) are not used, because their kernels are not
  bit-reproducible across runs.
"""
from __future__ import annotations

import os


def workers() -> int:
    env = os.environ.get("RESFED_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, os.cpu_count() or 1)


def deterministic() -> None:
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


deterministic()
