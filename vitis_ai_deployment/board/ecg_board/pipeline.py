"""int16 noisy window in -> int16 denoised window out, with per-stage timing."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from . import preprocess


class PipelineResult(NamedTuple):
    samples: np.ndarray  # int16[512], same units as the input
    pre_s: float
    infer_s: float
    post_s: float
    clipped: int


class DenoisePipeline(object):
    def __init__(self, backend, normalization: str = preprocess.DEFAULT_METHOD,
                 percentile: float = preprocess.DEFAULT_PERCENTILE):
        if normalization not in preprocess.METHODS:
            raise ValueError("normalization must be one of %s" % (preprocess.METHODS,))
        self.backend = backend
        self.normalization = normalization
        self.percentile = percentile

    def process(self, samples: np.ndarray) -> PipelineResult:
        t0 = time.perf_counter()
        raw = np.asarray(samples)
        if raw.shape != (preprocess.WINDOW_LENGTH,):
            raise ValueError("Expected %d samples, got shape %s" % (preprocess.WINDOW_LENGTH, raw.shape))
        normalized, affine = preprocess.normalize(raw.astype(np.float32), self.normalization, self.percentile)
        t1 = time.perf_counter()
        output = self.backend.infer(normalized)
        t2 = time.perf_counter()
        restored, clipped = preprocess.to_int16(preprocess.invert_affine(output, affine))
        t3 = time.perf_counter()
        return PipelineResult(restored, t1 - t0, t2 - t1, t3 - t2, clipped)

    def info(self) -> dict:
        return {"normalization": self.normalization, "percentile": self.percentile}
