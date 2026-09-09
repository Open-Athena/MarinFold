"""Benchmark TF32-enabled matmuls against the float32 reference sampler.

Use a separate output prefix and matched seeds. Compare coordinates and quality
before adopting this numerical setting in the diversity pilot.
"""

import torch

from sample_worker import main

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
