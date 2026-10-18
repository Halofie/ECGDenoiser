# PRD: Classical DSP Denoising Filter (Track A Baseline)

**Project Name:** ECG-Denoise-DSP-Baseline
**Relationship to project:** Non-learned counterpart to the FPGA-targeted AI
denoiser specified in [PRD-ECG-Denoising-Autoencoder-FPGA.md](PRD-ECG-Denoising-Autoencoder-FPGA.md)
("Track B"). Track A requires no training data, no accelerator, and no
quantization — it is the signal-processing baseline every learned model in
this project must beat to justify its cost.
**Primary Dataset:** PhysioNet MIT-BIH Arrhythmia Database (clean) mixed with
MIT-BIH Noise Stress Test Database records `em` (electrode motion / EM-like
noise) and `bw` (baseline wander) — identical dataset and noise-mixing
contract used by Track B.
**Document Purpose:** Requirements and evaluation contract for a classical
DSP filter chain used as the comparison baseline against the trained
autoencoder / ResUNet1D.

---

## 1. Executive Summary & Objective

Implement a deterministic, non-learned DSP filter pipeline ("Track A") that
removes baseline wander and high-frequency electrode-motion/EMG-like noise
from 360 Hz ECG signals, using only classical signal-processing operators
(median filtering, IIR Butterworth filtering, IIR notch filtering). Evaluate
it on the exact same held-out windows, noise mix, and target-SNR draws used
to evaluate the AI model, using the project's existing MSE / SNR-improvement
/ PRD metric suite, and report the two approaches side by side.

Track A exists to answer one question quantitatively: **how much of the
noise-removal problem does the trained network solve that a zero-training,
zero-parameter DSP chain does not?**

---

## 2. Why a DSP Baseline (not a hardware constraint document)

Unlike the AI-model PRD, Track A has no FPGA/HLS synthesis constraints — it
is deliberately a software-only reference implementation. Constraints here
are about **fairness of comparison**, not resource budgets:

| Constraint | Requirement | Rationale |
| :--- | :--- | :--- |
| **Input contract** | Same noisy/clean window pairs, same normalization, same target-SNR draw per window index as the AI model's evaluation set | Any accuracy delta must come from the algorithm, not from an easier test set |
| **No learned parameters** | Filter cutoffs and orders are fixed constants, chosen once from ECG physiology, never fit to this dataset | A DSP baseline that is tuned on the test set is not a fair "no training data" comparison |
| **No look-ahead beyond the window** | Filters operate causally or zero-phase within the same 512-sample (or 256-sample) window the model receives | Matches the model's available context |
| **Deterministic** | Same input always produces the same output | Required for reproducible comparison reports |

---

## 3. Signal Chain

```
Noisy window x[n]  (360 Hz, physical units, pre-normalization)
  │
  ├── Stage 1: Zero-phase Butterworth band-pass (order 2, 0.5-40 Hz)
  │     standard clinical-monitor front end; low edge rejects baseline
  │     wander, high edge rejects EMG-like/electrode-motion content
  │     implemented as SOS + sosfiltfilt (falls back to causal sosfilt
  │     if the window is too short for the default zero-phase padding)
  │
  ├── Stage 2 (optional, default on): 60 Hz IIR notch (Q = 30)
  │     suppresses mains-frequency interference, present for completeness
  │     even though nstdb em/bw records are not mains-tone-dominated
  │
  ▼
Denoised output x̂[n]
```

The filter runs on the **physical (pre-normalization) window**, then the
result is passed through the same clean-window-derived affine transform used
for the clean/noisy pair, before computing metrics. Applying it after the
per-window `[-1, 1]` normalization instead would filter a signal whose
non-zero mean is an artifact of that window's own affine anchor rather than
a physical DC offset, which a band-pass filter cannot distinguish from real
drift — an earlier iteration of this harness made that mistake and produced
a filter that appeared to actively damage the signal; see §5 for the
corrected numbers.

Implementation: [src/baseline/filters.py](src/baseline/filters.py)
(`dsp_denoise_filter`, plus the individual stages
`butterworth_lowpass_filter`, `butterworth_highpass_filter`,
`butterworth_bandpass_filter`, `notch_filter`, and
`median_baseline_removal` for ablation).

### Why a plain band-pass, not a median-filter baseline-wander cascade
A cascaded median filter (short/long stage, Pan-Tompkins-style) is a
well-known baseline-wander estimator for **continuous, multi-beat**
recordings. On this project's short evaluation windows (256-512 samples,
0.7-1.4 s), the long stage's ~0.6 s kernel is comparable to a single
heartbeat period, so it measurably removed QRS/T-wave morphology along with
the drift and made results worse, not better (empirically verified during
development). It is kept in the module for continuous-signal use cases but
is intentionally excluded from the default `dsp_denoise_filter` composite
used for this comparison.

### Why a band-pass rather than a matched notch bank for `em`
The MIT-BIH NSTDB `em` record is broadband electrode-motion/muscle-artifact
noise, not a narrowband tone, and its energy overlaps the diagnostic ECG
band itself (0.5-40 Hz) rather than sitting cleanly outside it. A band-pass
filter at the conventional clinical-monitor edges is the standard classical
response to this noise class; a notch filter is retained separately because
real deployments also see mains interference, which the nstdb records do not
model.

---

## 4. Evaluation & Comparison Metrics

Identical metric suite to the AI-model PRD, computed with the same
implementations (`src/training/metrics.py`, `src/evaluation/evaluate.py`) so
numbers are directly comparable:

1. **MSE** — `mean((clean - pred) ** 2)`
2. **SNR (output)** — `10 * log10(sum(clean**2) / sum((clean - pred)**2))` dB
3. **PRD** — `sqrt(sum((clean - pred)**2) / sum(clean**2)) * 100%` (target `< 5%` per the AI-model PRD)

In addition, Track A vs Track B are compared on:

4. **Trainable parameters** — 0 for the DSP filter vs. the AI model's
   parameter count.
5. **Training data requirement** — none for the DSP filter vs. a supervised
   clean/noisy training set for the AI model.
6. **Per-window CPU latency** (mean, p95) — both are cheap enough to run in
   real time on a CPU; this quantifies the actual gap, which matters because
   Track B additionally needs an FPGA/DPU deployment path that Track A does
   not.

Comparison harness: [scripts/compare_dsp_vs_ai.py](scripts/compare_dsp_vs_ai.py).
It runs both filters on the same held-out validation windows used by the
currently-checkpointed AI model (`checkpoints/ecg_resunet1d.pt`,
`ECGResUNet1D`, 512-sample windows, records `222/230/231/233/234`) and
writes:
- `reports/dsp_vs_ai_comparison.json` — full machine-readable results
- `reports/dsp_vs_ai_comparison.md` — human-readable overall + per-record tables

---

## 5. Current Result (12,690 held-out windows, `checkpoints/ecg_resunet1d.pt`)

| Approach | Mean MSE | Mean PRD | Mean SNR | Trainable params | Mean CPU latency/window |
|---|---:|---:|---:|---:|---:|
| Noisy input (no filtering) | 0.0664 | 46.61% | 9.02 dB | — | — |
| DSP filter (Track A) | 0.1140 | 62.22% | 5.63 dB | 0 | 3.00 ms |
| AI model (Track B) | 0.0329 | 36.39% | 9.87 dB | 360,321 | 24.23 ms |

**The classical band-pass/notch DSP filter does not beat doing nothing on
this dataset** — it makes PRD worse than the unfiltered noisy signal on
every one of the five held-out records. The AI model improves on the noisy
baseline (PRD 46.6% → 36.4%) and clearly outperforms the DSP filter.

This is the expected, literature-consistent outcome, not a defect in the
comparison: MIT-BIH NSTDB `em`/`bw` noise carries substantial energy inside
the 0.5-40 Hz diagnostic ECG band, so a linear frequency-selective filter
cannot separate noise from signal — removing the noise-carrying frequencies
also removes ECG morphology, and its own filtering artifacts (edge
transients, imperfect stopband attenuation) add further error on top. This
overlap is precisely the reason nonlinear, learned denoisers are studied for
this noise class instead of assuming a fixed linear filter is "good enough."
The Track B model's checkpoint here is also an interrupted run (18/60
epochs, `runs/ecg_resunet1d_progress.json`) — its PRD margin over both the
noisy baseline and the DSP filter should grow, not shrink, once training is
completed.

Reproduce with:
```powershell
python scripts/compare_dsp_vs_ai.py --checkpoint checkpoints/ecg_resunet1d.pt
```

---

## 6. Acceptance Criteria

- Filter functions are pure, deterministic, unit-tested (`tests/test_baseline_filters.py`), and operate on 1D `float32` NumPy arrays without raising on the project's 256- or 512-sample window lengths.
- The comparison script runs end-to-end against the currently checkpointed AI model with no dataset regeneration required.
- The resulting report states, per record and overall, whether the DSP filter or the AI model achieves lower PRD, and by how much.
- No claim of "the AI model is better" is accepted without this report backing it — the DSP baseline is the null hypothesis the learned model must beat.

---

## 7. Non-Goals

- No wavelet-transform baseline in this iteration (would require adding
  `pywt` as a new dependency; the config's `evaluation.baselines` list keeps
  `wavelet` as a documented future baseline option, not implemented here).
- No adaptive filtering (LMS/RLS) baseline — out of scope for this PRD.
- No FPGA/HLS export of the DSP chain — it is a software reference only,
  used to quantify the value the learned model adds before that model is
  deployed to hardware.
