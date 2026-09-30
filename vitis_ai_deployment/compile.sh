#!/usr/bin/env bash
# Compile the PTQ/QAT-exported xmodel into a DPU-deployable xmodel for ZCU104.
#
# Run inside the Vitis AI 3.5 tools Docker image (the same one used for
# quantize.py / train.py --mode qat), NOT on the board.
#
# IMPORTANT -- arch.json path correction:
# Vitis AI 1.x / DNNDK used paths like /arch/dpuv2/ZCU104/arch.json. Vitis AI
# 3.5 targets the DPUCZDX8G IP and ships arch.json files under the compiler
# package instead:
#
#   /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
#
# If your ZCU104 image was built from a custom DPU TRD configuration (a
# different DPU fingerprint/frequency/RAM usage than the stock ZCU104
# reference design), you must use the arch.json generated for THAT specific
# DPU build instead of the stock one, or the compiled xmodel will not match
# the DPU actually programmed into the PL. Confirm the fingerprint with:
#
#   xdputil query
#
# run on the board against the loaded DPU overlay, and compare it against
# `xir dump_txt <xmodel>` metadata after compiling, before trusting the
# match. This has not been verified in this workspace -- no Vitis AI
# install or ZCU104 board is available here.

set -euo pipefail

XMODEL="${1:-artifacts/quantize_result/ECGResUNet1D_int.xmodel}"
ARCH_JSON="${2:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json}"
OUTPUT_DIR="${3:-./compiled_model}"
NET_NAME="${4:-ecg_denoiser}"

if [ ! -f "$XMODEL" ]; then
    echo "xmodel not found: $XMODEL" >&2
    echo "Run quantize.py --mode test (or train.py --mode qat) first." >&2
    exit 1
fi

if [ ! -f "$ARCH_JSON" ]; then
    echo "arch.json not found: $ARCH_JSON" >&2
    echo "Locate the correct arch.json for your Vitis AI 3.5 install / ZCU104 DPU build." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

vai_c_xir \
    --xmodel "$XMODEL" \
    --arch "$ARCH_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --net_name "$NET_NAME"

echo "Compiled xmodel written to $OUTPUT_DIR/${NET_NAME}.xmodel"
echo "Copy this file to the ZCU104 (e.g. via scp) and use it with deploy_zcu104.py."
