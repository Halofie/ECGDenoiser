"""Board-side runtime for the ECG denoiser DPU deployment on the ZCU104.

Everything in this package must import on the ZCU104 Vitis AI board image,
which has numpy + vart/xir but NOT torch, scipy, or this repo's src/ tree.
Keep it numpy-only and Python 3.7 compatible (no walrus, no match, no
runtime PEP 604/585 annotations -- `from __future__ import annotations` is
fine). The host-side scripts import these same modules so the protocol and
pre/post-processing can never drift between laptop and board.
"""

__version__ = "1.0.0"
