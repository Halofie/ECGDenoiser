# PRD: FPGA-Optimized 1D Autoencoder for ECG Electromagnetic Denoising

**Project Name:** ECG-Denoise-FPGA-1DCAE  
**Target Architecture:** Xilinx Zynq-7000 / Zynq UltraScale+ (PYNQ Platform)  
**Primary Dataset:** PhysioNet MIT-BIH Noise Stress Test Database (`nstdb` v1.0.0)  
**Conversion Framework:** `hls4ml` (or Xilinx FINN)  
**Document Purpose:** System Requirements and Machine Learning Specification for Automated Code Generation / GitHub Copilot Task Execution.

---

## 1. Executive Summary & Objective
The goal of this project is to implement an end-to-end Machine Learning pipeline that trains a Quantization-Aware 1D Convolutional Autoencoder (1D-CAE) to remove Electromagnetic (EM) interference and baseline wander from raw ECG signals. The resulting model must be directly exportable to High-Level Synthesis (HLS) C++ for FPGA deployment with minimal latency and deterministic resource usage.

---

## 2. Hardware Constraints & Synthesis Rules
To ensure the generated neural network is synthesizable on Xilinx/PYNQ FPGA architectures without LUT/DSP overflow, all model implementations **must** abide by the following strict rules:

| Constraint | Allowed / Requirement | Forbidden / Anti-Pattern | Hardware Justification |
| :--- | :--- | :--- | :--- |
| **Operators** | `Conv1D`, `MaxPool1D`, `UpSample1D` (Nearest), `BatchNorm1D`, `Dense` | RNNs (`LSTM`, `GRU`), Self-Attention, Dynamic Loops | Sequential dependencies destroy pipeline parallelism. |
| **Activations** | `ReLU` | `Sigmoid`, `Tanh`, `GELU`, `Swish`, `Softmax` | Transcendentals require huge LUT lookups or floating-point units. |
| **Downsampling** | Strided `Conv1D` (`stride=2`) or `MaxPool1D` | `AveragePooling1D` (non-power-of-2 div) | Avoids hardware division units. |
| **Channel Sizes** | Powers of 2 ($16, 32, 64$) | Arbitrary filter counts (e.g., $17, 23$) | Aligns directly with BRAM memory bus widths (AXI-Stream). |
| **Normalization** | `BatchNorm1D` (Foldable during inference) | LayerNorm, InstanceNorm | BatchNorm parameters can be mathematically merged into Conv weights post-training. |
| **Precision** | INT8 / Fixed-point (`ap_fixed<8,3>`) via QAT | Standard FP32 / FP64 | Pure integer arithmetic utilizes native DSP slices efficiently. |

---

## 3. Data Pipeline & Signal Preprocessing

### 3.1 Datasets
1. **Clean Signal:** PhysioNet MIT-BIH Normal Sinus Rhythm Database (`nsrdb`) or MIT-BIH Arrhythmia Database (`mitdb`).
2. **Noise Signal:** PhysioNet MIT-BIH Noise Stress Test Database (`nstdb`), specifically the `em` (Electromagnetic noise) and `bw` (Baseline wander) records.

### 3.2 Preprocessing Specifications
* **Window Length ($N$):** Exactly $N = 256$ contiguous samples (Sampling rate $f_s = 360\text{ Hz}$).
* **Dynamic SNR Noise Injection:**
  $$\text{ECG}_{noisy}[n] = \text{ECG}_{clean}[n] + \alpha \cdot \text{Noise}_{EM}[n]$$
  Where $\alpha$ is computed on the fly during data loading to target random SNRs in the range $[-5\text{ dB}, 15\text{ dB}]$.
* **Normalization:** Per-window Min-Max scaling to range $[-1, 1]$:
  $$\mathbf{x}_{norm} = 2 \cdot \frac{\mathbf{x} - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})} - 1$$

---

## 4. Model Architecture (1D-CAE with QAT)

### 4.1 Topology Map
The architecture must be implemented using **Brevitas** (PyTorch) or **QKeras** (TensorFlow/Keras). For Vitis HLS 2023.2, the deployment model uses only operators supported by the hls4ml PyTorch frontend. The model must be retrained from scratch after this architecture change.

```
Input Vector (1 x 256)
  │
  ├── [QuantConv1D]  16 Filters, Kernel=7, Stride=1, Padding=3  ──► (16 x 256)
  ├── [QuantReLU]
  │
  ├── [QuantConv1D]  32 Filters, Kernel=5, Stride=1, Padding=2  ──► (32 x 256)
  ├── [QuantReLU]
  │
  ├── [QuantConv1D]  32 Filters, Kernel=3, Stride=1, Padding=1  ──► (32 x 256)
  ├── [QuantReLU]  ─── [ BOTTLENECK: 32 Channels x 256 Spatial ]
  │
  ├── [QuantConv1D]  16 Filters, Kernel=3, Stride=1, Padding=1  ──► (16 x 256)
  ├── [QuantReLU]
  │
  ├── [QuantConv1D]  8 Filters, Kernel=5, Stride=1, Padding=2   ──► (8 x 256)
  ├── [QuantReLU]
  │
  ├── [QuantConv1D]  1 Filter,   Kernel=7, Stride=1, Padding=3  ──► Output (1 x 256)
```

All layers preserve the 256-sample spatial length. The architecture intentionally avoids strided convolution, upsampling, dynamic shape operations, and skip connections because these are not supported reliably by the Vitis HLS 2023.2 hls4ml frontend.

### 4.2 Quantization Specifications
* **Weight Quantization:** 8-bit signed integer (`int8` / `ap_fixed<8,1>`).
* **Activation Quantization:** 8-bit unsigned/signed integer (`uint8` / `ap_fixed<8,2>`).
* **Accumulator Bitwidth:** 24-bit or 32-bit to prevent MAC overflow.

---

## 5. Evaluation & Performance Metrics

The model must be validated using three core signal quality metrics comparing ground truth clean signal $x[n]$ against reconstructed signal $\hat{x}[n]$:

1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} (x[n] - \hat{x}[n])^2$$

2. **Signal-to-Noise Ratio (SNR) Improvement:**
   $$\text{SNR}_{out} = 10 \log_{10} \left( \frac{\sum_{n=1}^{N} x[n]^2}{\sum_{n=1}^{N} (x[n] - \hat{x}[n])^2} \right)$$

3. **Percentage Root-Mean-Square Difference (PRD) [Clinical Metric]:**
   $$\text{PRD} = \sqrt{ \frac{\sum_{n=1}^{N} (x[n] - \hat{x}[n])^2}{\sum_{n=1}^{N} x[n]^2} } \times 100\%$$
   *Target: $\text{PRD} < 5\%$ for high-fidelity ECG diagnostic preservation.*

---

## 6. Implementation Roadmap for Copilot / Engineer

### Task 1: Dataset Loader (`data_loader.py`)
- Write a script using `wfdb` to download MIT-BIH Arrhythmia (`mitdb`) and Noise Stress Test (`nstdb`).
- Construct a PyTorch `Dataset` / TensorFlow `Sequence` class that reads raw files, extracts 256-sample segments, mixes clean signals with `em` noise at dynamically calculated SNRs, and normalizes outputs.

### Task 2: Model Definition (`model.py`)
- Implement the 1D Quantized Autoencoder in **Brevitas** or **QKeras**.
- Verify that every layer utilizes quantized convolutions and activations.

### Task 3: Training & Evaluation (`train.py`)
- Train using Adam optimizer ($lr=10^{-3}$), MSE loss, for 50 epochs.
- Calculate and log average MSE, SNR improvement, and PRD metrics over the test set.

### Task 4: HLS Conversion Script (`export_hls.py`)
- Write an `hls4ml` conversion script to translate the trained model into Vivado HLS code.
- Configure `hls4ml` strategy:
  * **Precision:** `ap_fixed<16,6>` or matching QAT specs.
  * **Reuse Factor:** Set to target desired latency vs. DSP usage.
- Compile and generate IP block.
