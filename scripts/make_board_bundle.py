"""Assemble everything the ZCU104 needs into one folder + tarball.

    python scripts/make_board_bundle.py \
        --xmodel vitis_ai_deployment/compiled_model_noisy_minmax/ecg_denoiser.xmodel

Produces dist/zcu104_bundle/ and dist/zcu104_bundle.tar.gz containing:
    ecg_board/            runtime package (numpy-only)
    uart_server.py        UART/TCP denoising service
    board_session.py      offline measurements (golden, latency, throughput, power)
    power_monitor.py      list/log power rails
    uart_setup.sh         find / free a UART
    run_all.sh            the board-day sequence in one script
    ecg_denoiser.xmodel   the compiled model
    golden_vectors/       input_*.npy + golden_meta.json (expected_*.npy stay on the laptop)
    MANIFEST.json         md5s, git commit, normalization -- proves what was deployed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "vitis_ai_deployment" / "board"

RUN_ALL = """#!/bin/sh
# Board-day sequence. Run on the ZCU104 from this folder:  sh run_all.sh
# Extra arguments are passed to board_session.py (e.g. --idle-seconds 30).
set -e
cd "$(dirname "$0")"
NORMALIZATION="{normalization}"

echo "=== 1. Environment + serial ports"
sh uart_setup.sh || true

echo "=== 2. Power rails"
python3 power_monitor.py list --write-map rails_template.json || true
RAIL_ARGS=""
if [ -f rails.json ]; then RAIL_ARGS="--rail-map rails.json"; fi

echo "=== 3. Offline session: golden vectors, latency, throughput, power"
python3 board_session.py --xmodel ecg_denoiser.xmodel --golden-dir golden_vectors \\
    --normalization "$NORMALIZATION" $RAIL_ARGS "$@"

echo
echo "=== 4. Next: start the UART service, then run the host client on the laptop"
echo "python3 uart_server.py --xmodel ecg_denoiser.xmodel --normalization $NORMALIZATION \\\\"
echo "    --serial /dev/ttyPS1 --baud 115200 --power $RAIL_ARGS"
"""


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xmodel", required=True, type=Path)
    parser.add_argument("--golden-dir", default=ROOT / "artifacts" / "golden_vectors", type=Path)
    parser.add_argument("--normalization", default="noisy_minmax", choices=("noisy_minmax", "noisy_robust"))
    parser.add_argument("--rail-map", default=None, type=Path, help="Optional rails.json to include.")
    parser.add_argument("--out", default=ROOT / "dist" / "zcu104_bundle", type=Path)
    args = parser.parse_args()

    if not args.xmodel.is_file():
        raise SystemExit(f"xmodel not found: {args.xmodel}")
    meta_path = args.golden_dir / "golden_meta.json"
    golden_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if golden_meta.get("normalization", args.normalization) != args.normalization:
        raise SystemExit(f"Golden vectors were made with {golden_meta['normalization']}, bundle uses "
                         f"{args.normalization}. Regenerate them (generate_golden_vectors.py).")
    if not golden_meta:
        print(f"WARNING: no {meta_path}; golden vectors may predate the deployable normalization.")

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(BOARD_DIR, args.out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results"))
    shutil.copy2(args.xmodel, args.out / "ecg_denoiser.xmodel")
    golden_out = args.out / "golden_vectors"
    golden_out.mkdir()
    inputs = sorted(args.golden_dir.glob("input_*.npy"))
    for path in inputs + ([meta_path] if meta_path.is_file() else []):
        shutil.copy2(path, golden_out / path.name)
    if args.rail_map:
        shutil.copy2(args.rail_map, args.out / "rails.json")
    (args.out / "run_all.sh").write_bytes(RUN_ALL.format(normalization=args.normalization).encode())

    xmodel_meta = args.xmodel.parent / "meta.json"
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "xmodel_source": str(args.xmodel),
        "xmodel_md5": md5(args.xmodel),
        "xmodel_meta": json.loads(xmodel_meta.read_text()) if xmodel_meta.is_file() else None,
        "normalization": args.normalization,
        "golden_vectors": len(inputs),
        "golden_meta": golden_meta,
        "files": {str(p.relative_to(args.out)).replace("\\", "/"): md5(p)
                  for p in sorted(args.out.rglob("*")) if p.is_file()},
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    archive = args.out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(args.out, arcname=args.out.name)
    print(f"Bundle: {args.out} ({len(manifest['files'])} files), xmodel md5 {manifest['xmodel_md5']}")
    print(f"Archive: {archive}")
    print(f"Copy to the board:  scp {archive} root@<board-ip>:~/  &&  ssh root@<board-ip> "
          f"'tar xzf {archive.name} && cd {args.out.name} && sh run_all.sh'")


if __name__ == "__main__":
    main()
