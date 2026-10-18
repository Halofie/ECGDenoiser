# DSP Filter (Track A) vs AI Model / ECGResUNet1D (Track B) Comparison

- Test windows: 12690 (held-out validation records, 512-sample windows)
- Sample rate: 360.0 Hz
- Val seed: 1000
- Checkpoint: `checkpoints/ecg_resunet1d.pt`

## Overall Quality Metrics

| Approach | Mean MSE | Mean PRD | Mean SNR |
|---|---:|---:|---:|
| Noisy input (no filtering) | 0.066384 | 46.6051% | 9.0201 dB |
| DSP filter (Track A) | 0.113984 | 62.2158% | 5.6256 dB |
| AI model (Track B) | 0.032881 | 36.3906% | 9.8690 dB |

## Efficiency / Deployment Comparison

| Metric | DSP filter (Track A) | AI model (Track B) |
|---|---:|---:|
| Trainable parameters | 0 | 360321 |
| Requires labeled training data | False | True |
| Mean latency / window (CPU) | 3.0035 ms | 24.2279 ms |
| p95 latency / window (CPU) | 4.2582 ms | 35.4332 ms |

## Per-Record Quality Metrics

| Record | Approach | Mean MSE | Mean PRD | Mean SNR |
|---|---|---:|---:|---:|
| 222 | Noisy | 0.093655 | 43.4069% | 9.3521 dB |
| 222 | DSP filter | 0.206389 | 70.2005% | 4.2575 dB |
| 222 | AI model | 0.042102 | 33.7958% | 10.4827 dB |
| 230 | Noisy | 0.048198 | 62.7586% | 6.1853 dB |
| 230 | DSP filter | 0.063673 | 78.0650% | 3.4838 dB |
| 230 | AI model | 0.019956 | 45.7267% | 8.0704 dB |
| 231 | Noisy | 0.051928 | 42.9445% | 9.6145 dB |
| 231 | DSP filter | 0.089499 | 56.8391% | 6.2755 dB |
| 231 | AI model | 0.031369 | 35.9725% | 9.3847 dB |
| 233 | Noisy | 0.073268 | 57.0004% | 6.8061 dB |
| 233 | DSP filter | 0.093549 | 67.5612% | 4.6607 dB |
| 233 | AI model | 0.031136 | 42.1202% | 8.8889 dB |
| 234 | Noisy | 0.064868 | 26.9153% | 13.1426 dB |
| 234 | DSP filter | 0.116809 | 38.4130% | 9.4504 dB |
| 234 | AI model | 0.039844 | 24.3377% | 12.5183 dB |
