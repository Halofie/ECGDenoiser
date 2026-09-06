# Checkpoint Evaluation Report

**Date:** 2026-09-04  
**Checkpoint:** `checkpoints/best_model.pt`  
**Checkpoint epoch:** 31  
**Status:** Interim checkpoint from an interrupted training run; not the completed 50-epoch experiment.

## Test Configuration

- Test records: `100`, `105`, `117`, `220`
- Test windows: 20,308
- Window length: 256 samples
- Window stride: 128 samples
- Device: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Input normalization: per-window min-max scaling to `[-1, 1]`
- Noise: weighted local EM/BW noise
- Target SNR range: `-5 dB` to `15 dB`

## Overall Results

| Metric | Noisy input baseline | Model output |
|---|---:|---:|
| Mean MSE | 0.1542226 | 0.0902949 |
| Mean PRD | 66.9913% | 52.7513% |
| Mean output SNR | 6.5260 dB | 8.8932 dB |

### Derived changes

- MSE reduction: approximately 41.5%
- PRD reduction: approximately 21.3%
- SNR improvement over noisy input: approximately +2.37 dB

## Per-Record Results

| Record | Windows | Model MSE | Model PRD | Model SNR |
|---|---:|---:|---:|---:|
| 100 | 5,077 | 0.0472115 | 27.3558% | 14.2279 dB |
| 105 | 5,077 | 0.0452665 | 27.0537% | 13.1453 dB |
| 117 | 5,077 | 0.1967089 | 104.1786% | 1.0066 dB |
| 220 | 5,077 | 0.0708066 | 52.2634% | 7.2458 dB |

## Efficiency Results

- Trainable parameters: 11,878
- FP32 parameter memory: approximately 47.5 KB
- Checkpoint size: approximately 107 KB
- Mean GPU inference time: approximately 0.872 ms per window
- Mean inference time per batch: approximately 55.689 ms
- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU

## Acceptance Assessment

The PRD target is PRD `< 5%`.

**Current model PRD: 52.7513%**

The current checkpoint does **not** meet the required accuracy target. It does improve the noisy baseline overall, but performance is inconsistent, especially on record `117`.

## Interpretation and Risks

- This is an interim checkpoint from epoch 31, not a final trained model.
- The 50-epoch run was stopped before completion.
- No morphology-preservation metrics were measured yet for QRS timing, R-peak agreement, or beat-level correlation.
- No Butterworth, notch, or wavelet baseline comparison has been completed.
- No FPGA HLS conversion, synthesis, resource, timing, or throughput report exists yet.
- Separate clean/noisy normalization may affect the effective SNR relationship after mixing.
- The current result should be used as a baseline for subsequent experiments, not as a final accuracy claim.

## Recommended Next Experiments

1. Complete a controlled training run and preserve its training history.
2. Compare shared-statistics normalization against separate normalization.
3. Analyze failure behavior on record `117`.
4. Add R-peak, QRS timing, and waveform-correlation evaluation.
5. Compare against the configured signal-processing baselines.
6. Measure FP32 versus quantized model degradation.
7. Export only after software accuracy and morphology checks are acceptable.
