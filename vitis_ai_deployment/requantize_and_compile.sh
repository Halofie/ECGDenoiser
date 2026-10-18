#!/usr/bin/env bash
# PTQ calibrate -> INT8 accuracy check -> export -> compile, for one checkpoint.
#
# Run INSIDE the Vitis AI Docker image (same image/version as the board's
# Vitis AI runtime -- an xmodel compiled by 3.5 tools will not load on a 3.0
# board image and vice versa), from this directory:
#
#   conda activate vitis-ai-pytorch          # env name in the 3.x PyTorch images
#   bash requantize_and_compile.sh ../checkpoints/ecg_resunet1d_noisy_minmax.pt
#
# Optional env overrides:
#   ARCH_JSON      default /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
#   NORMALIZATION  default noisy_minmax (must match the checkpoint; checked)
#   TAG            default <normalization>; names the output folders
#   EVAL_WINDOWS   default 500 (held-out windows for the float vs INT8 check)
#
# Outputs:
#   artifacts/quantize_result_<TAG>/ECGResUNet1D_int.xmodel + quant_eval.json
#   compiled_model_<TAG>/ecg_denoiser.xmodel  (+ md5sum.txt, meta.json)

set -euo pipefail

CHECKPOINT="${1:?usage: bash requantize_and_compile.sh <checkpoint.pt>}"
ARCH_JSON="${ARCH_JSON:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json}"
NORMALIZATION="${NORMALIZATION:-noisy_minmax}"
TAG="${TAG:-$NORMALIZATION}"
EVAL_WINDOWS="${EVAL_WINDOWS:-500}"
QUANT_DIR="artifacts/quantize_result_${TAG}"
COMPILED_DIR="compiled_model_${TAG}"
DATA_ARGS=(--clean-dir ../data/clean --noise-dir ../mit-bih-noise-stress-test-database-1.0.0)

echo "== 1/3 calibrate ($CHECKPOINT, normalization=$NORMALIZATION)"
python quantize.py --mode calib --checkpoint "$CHECKPOINT" --normalization "$NORMALIZATION" \
    --output-dir "$QUANT_DIR" "${DATA_ARGS[@]}"

echo "== 2/3 INT8 check + export"
python quantize.py --mode test --checkpoint "$CHECKPOINT" --normalization "$NORMALIZATION" \
    --output-dir "$QUANT_DIR" --evaluate "$EVAL_WINDOWS" "${DATA_ARGS[@]}"

echo "== 3/3 compile for $(basename "$(dirname "$ARCH_JSON")")"
bash compile.sh "$QUANT_DIR/ECGResUNet1D_int.xmodel" "$ARCH_JSON" "$COMPILED_DIR" ecg_denoiser

echo
echo "Compiled: $COMPILED_DIR/ecg_denoiser.xmodel"
cat "$COMPILED_DIR/meta.json" 2>/dev/null || true
echo
echo "Quantization check ($QUANT_DIR/quant_eval.json):"
python - "$QUANT_DIR/quant_eval.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for k in ("noisy_input", "float", "int8_fake_quant"):
    print("  %-16s PRD %6.2f%%" % (k, r[k]["prd_percent"]))
delta = r["int8_minus_float_prd_points"]
print("  INT8 - float: %+.2f PRD points%s" % (delta, "  <-- investigate before board day" if delta > 2 else ""))
PY
echo "Check the compile log above: it must report exactly ONE DPU subgraph."
