"""Stream noisy ECG to the ZCU104 over UART (or TCP), collect denoised output, score it.

Two modes:

  windows     Held-out validation windows (records 222/230/231/233/234, seed
              1000 -- the same windows as every other evaluation in reports/).
              Scores board output vs the clean signal, and vs the laptop's own
              float model on the identical int16 input (the correctness check).
  continuous  One record, mixed with noise at a fixed SNR, streamed as
              overlapping windows (hop 256) and stitched back together with
              overlap-add. Optional --realtime pacing sends one window per
              hop (0.711 s at 360 Hz) to show the link keeps up live.

Examples:
    # board day (Windows COM port of the ZCU104 USB-UART)
    python scripts/uart_host_client.py --port COM7 --baud 115200 --num-windows 1000
    python scripts/uart_host_client.py --port COM7 --mode continuous --record 231 --seconds 60 --realtime
    # rehearsal against `uart_server.py --backend torch --tcp-port 5555`
    python scripts/uart_host_client.py --port tcp://127.0.0.1:5555 --num-windows 200

Wire format: int16 counts, --gain counts per mV (default 2000 => +/-16.4 mV full
scale; the validation set peaks at 5.7 mV). The board's normalization is
scale-free, so the gain only sets resolution/headroom.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VITIS_DIR = ROOT / "vitis_ai_deployment"
BOARD_DIR = VITIS_DIR / "board"
for path in (ROOT, VITIS_DIR, BOARD_DIR):
    sys.path.insert(0, str(path))

from ecg_board import preprocess, protocol  # noqa: E402
from ecg_board.client import BoardClient  # noqa: E402
from ecg_board.common import latency_stats  # noqa: E402
from ecg_board.pipeline import DenoisePipeline  # noqa: E402
from ecg_board.transport import open_transport  # noqa: E402

SAMPLE_RATE_HZ = 360.0
HOP = 256
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "ecg_resunet1d_noisy_minmax.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", required=True, help="COM7, /dev/ttyUSB1, or tcp://HOST:PORT")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=3.0, help="Reply timeout per attempt (s).")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--mode", choices=("windows", "continuous"), default="windows")
    parser.add_argument("--num-windows", type=int, default=500, help="windows mode; 0 = all 12,690.")
    parser.add_argument("--record", default="231", help="continuous mode record.")
    parser.add_argument("--seconds", type=float, default=60.0, help="continuous mode duration.")
    parser.add_argument("--snr-db", type=float, default=5.0, help="continuous mode input SNR.")
    parser.add_argument("--realtime", action="store_true", help="Pace sends at the ECG rate (one hop per window).")
    parser.add_argument("--gain", type=float, default=2000.0, help="int16 counts per mV on the wire.")
    parser.add_argument("--reference-checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="Laptop float model run on the same int16 input; '' to skip.")
    parser.add_argument("--normalization", choices=preprocess.METHODS, default=preprocess.DEFAULT_METHOD)
    parser.add_argument("--clean-dir", default=str(ROOT / "data" / "clean"))
    parser.add_argument("--noise-dir", default=str(ROOT / "mit-bih-noise-stress-test-database-1.0.0"))
    parser.add_argument("--out-dir", default=None, help="Default: results/host_<mode>_<timestamp>.")
    parser.add_argument("--no-bye", action="store_true", help="Leave the board server running afterwards.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- data

def load_windows(args):
    """Return list of dicts with clean/noisy physical windows (mV) + provenance."""
    from dataset import ECG512Dataset, default_train_val_split

    _, val_records = default_train_val_split()
    dataset = ECG512Dataset(args.clean_dir, args.noise_dir, records=val_records,
                            deterministic=True, seed=1000, download_missing=False)
    count = len(dataset) if args.num_windows <= 0 else min(args.num_windows, len(dataset))
    items = []
    for index in np.linspace(0, len(dataset) - 1, count, dtype=int):
        clean, noisy = dataset.raw_pair(int(index))
        sample = dataset._samples[int(index)]
        items.append({"index": int(index), "record": sample["record"], "start": sample["start"],
                      "clean": np.asarray(clean, np.float32), "noisy": np.asarray(noisy, np.float32)})
    return items


def load_continuous(args):
    """One record segment + noise at a fixed SNR, as (clean, noisy) physical signals."""
    from dataset import NOISE_WEIGHTS, combine_noise_signals, load_noise_records
    from src.data.fetch_dataset import load_clean_record
    from src.noise.noise_mix import mix_clean_and_noise

    record = load_clean_record(Path(args.clean_dir) / args.record)
    length = int(args.seconds * SAMPLE_RATE_HZ)
    length = max(512, (length - 512) // HOP * HOP + 512)
    clean = np.asarray(record["signal"][:length], np.float32)
    noise = combine_noise_signals(load_noise_records(args.noise_dir), NOISE_WEIGHTS, clean.size)
    noisy, _alpha = mix_clean_and_noise(clean, noise, args.snr_db)
    return clean, np.asarray(noisy, np.float32)


def to_wire(signal_mv: np.ndarray, gain: float):
    return preprocess.to_int16(np.asarray(signal_mv, np.float64) * gain)


def clean_domain(values_mv: np.ndarray, clean_mv: np.ndarray) -> np.ndarray:
    low, high = float(clean_mv.min()), float(clean_mv.max())
    return 2.0 * (values_mv - low) / max(high - low, 1e-6) - 1.0


def score(clean_mv: np.ndarray, pred_mv: np.ndarray) -> dict:
    """PRD/SNR/MSE in the clean window's [-1, 1] scale -- same convention as reports/."""
    clean_n, pred_n = clean_domain(clean_mv, clean_mv), clean_domain(pred_mv, clean_mv)
    err = float(np.sum((clean_n - pred_n) ** 2))
    energy = max(float(np.sum(clean_n ** 2)), 1e-12)
    return {"prd": float(np.sqrt(err / energy) * 100.0), "snr_db": float(10 * np.log10(energy / max(err, 1e-12))),
            "mse": float(np.mean((clean_n - pred_n) ** 2))}


# ----------------------------------------------------------------------------- run

def make_reference(args):
    if not args.reference_checkpoint:
        return None
    path = Path(args.reference_checkpoint)
    if not path.is_file():
        print(f"Reference checkpoint {path} not found; skipping laptop reference.")
        return None
    from ecg_board.backends import TorchBackend

    backend = TorchBackend(str(path))
    if backend.normalization != args.normalization:
        print(f"WARNING: reference checkpoint trained with {backend.normalization}, "
              f"streaming with {args.normalization}.")
    return DenoisePipeline(backend, args.normalization)


def stream(client: BoardClient, windows_int16, pace_s: float | None, on_reply=None):
    replies = []
    next_send = time.perf_counter()
    for i, samples in enumerate(windows_int16):
        if pace_s:
            delay = next_send - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_send += pace_s
        reply = client.denoise(samples)
        replies.append(reply)
        if on_reply:
            on_reply(i, reply)
    return replies


def timing_rows(replies, baud: int | None) -> tuple[list[dict], dict]:
    wire_ms = (protocol.REQUEST_SIZE + protocol.RESPONSE_SIZE) * 10 / baud * 1e3 if baud else None
    rows = []
    for i, r in enumerate(replies):
        rows.append({"window": i, "rtt_ms": r.round_trip_s * 1e3, "board_ms": r.timing.board_us / 1e3,
                     "infer_ms": r.timing.infer_us / 1e3, "pre_ms": r.timing.pre_us / 1e3,
                     "post_ms": r.timing.post_us / 1e3, "attempts": r.attempts, "clipped": r.timing.clipped,
                     "link_and_host_ms": r.round_trip_s * 1e3 - r.timing.board_us / 1e3})
    clean_rows = [row for row in rows if row["attempts"] == 1]  # retries distort the RTT
    summary = {
        "windows": len(rows),
        "retried_windows": len(rows) - len(clean_rows),
        "theoretical_wire_ms": wire_ms,
        "round_trip": latency_stats(r["rtt_ms"] / 1e3 for r in clean_rows),
        "board_total": latency_stats(r["board_ms"] / 1e3 for r in clean_rows),
        "board_infer": latency_stats(r["infer_ms"] / 1e3 for r in clean_rows),
        "board_pre": latency_stats(r["pre_ms"] / 1e3 for r in clean_rows),
        "board_post": latency_stats(r["post_ms"] / 1e3 for r in clean_rows),
        "link_and_host": latency_stats(r["link_and_host_ms"] / 1e3 for r in clean_rows),
    }
    return rows, summary


def run_windows(args, client, reference, out_dir: Path) -> dict:
    items = load_windows(args)
    wire = [to_wire(item["noisy"], args.gain) for item in items]
    clipped_inputs = sum(c for _, c in wire)
    print(f"Streaming {len(items)} held-out windows ({clipped_inputs} input samples clipped to int16)...")

    def progress(i, reply):
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)} rtt {reply.round_trip_s * 1e3:.1f} ms "
                  f"(board {reply.timing.board_us / 1e3:.2f} ms, infer {reply.timing.infer_us / 1e3:.2f} ms)")

    replies = stream(client, [w for w, _ in wire], 1.0 / SAMPLE_RATE_HZ * HOP if args.realtime else None, progress)
    rows, timing = timing_rows(replies, args.baud if not args.port.startswith("tcp://") else None)

    board_out = np.stack([r.samples for r in replies])
    ref_out = np.stack([reference.process(w).samples for w, _ in wire]) if reference else None
    quality = {"noisy": [], "board": [], "reference": []}
    agreement = []
    for i, item in enumerate(items):
        quality["noisy"].append(score(item["clean"], wire[i][0] / args.gain))
        quality["board"].append(score(item["clean"], board_out[i] / args.gain))
        rows[i].update({"record": item["record"], "start": item["start"],
                        "prd_noisy": quality["noisy"][-1]["prd"], "prd_board": quality["board"][-1]["prd"]})
        if ref_out is not None:
            quality["reference"].append(score(item["clean"], ref_out[i] / args.gain))
            diff = (board_out[i].astype(np.float64) - ref_out[i]) / args.gain
            span = max(float(np.ptp(ref_out[i])) / args.gain, 1e-6)
            agreement.append({"rmse_mv": float(np.sqrt(np.mean(diff ** 2))), "max_abs_mv": float(np.abs(diff).max()),
                              "rmse_pct_of_span": float(np.sqrt(np.mean(diff ** 2)) / span * 100),
                              "corr": float(np.corrcoef(board_out[i], ref_out[i])[0, 1])
                              if np.std(board_out[i]) > 0 and np.std(ref_out[i]) > 0 else float("nan")})
            rows[i].update({"prd_reference": quality["reference"][-1]["prd"], **{f"agree_{k}": v for k, v in agreement[-1].items()}})

    np.savez_compressed(out_dir / "host_windows.npz", noisy_int16=np.stack([w for w, _ in wire]),
                        board_int16=board_out, reference_int16=ref_out if ref_out is not None else np.zeros(0),
                        clean_mv=np.stack([i["clean"] for i in items]), gain=args.gain,
                        records=np.array([i["record"] for i in items]), starts=np.array([i["start"] for i in items]))
    write_csv(out_dir / "host_windows.csv", rows)

    def mean_of(entries):
        return {k: float(np.mean([e[k] for e in entries])) for k in entries[0]} if entries else None

    summary = {
        "mode": "windows",
        "num_windows": len(items),
        "input_clipped_samples": int(clipped_inputs),
        "quality": {k: mean_of(v) for k, v in quality.items()},
        "board_vs_reference": {
            "mean": mean_of(agreement),
            "worst_rmse_pct_of_span": float(max(a["rmse_pct_of_span"] for a in agreement)) if agreement else None,
        },
        "timing": timing,
    }
    if not args.no_plots:
        plot_windows(out_dir, items, wire, board_out, ref_out, rows, args.gain)
    return summary


def run_continuous(args, client, reference, out_dir: Path) -> dict:
    clean, noisy = load_continuous(args)
    noisy_wire, clipped = to_wire(noisy, args.gain)
    starts = list(range(0, noisy_wire.size - 512 + 1, HOP))
    print(f"Streaming record {args.record}: {noisy_wire.size / SAMPLE_RATE_HZ:.1f} s, {len(starts)} windows, "
          f"SNR {args.snr_db} dB{' (real-time pacing)' if args.realtime else ''}...")
    replies = stream(client, [noisy_wire[s:s + 512] for s in starts], HOP / SAMPLE_RATE_HZ if args.realtime else None)
    rows, timing = timing_rows(replies, args.baud if not args.port.startswith("tcp://") else None)

    def stitch(outputs):
        # Periodic Hann windows at 50% overlap sum to exactly 1; the outer half
        # of the first and last window is flat so the signal ends keep full weight.
        hann = np.hanning(513)[:512]
        acc = np.zeros(noisy_wire.size)
        for k, (s, out) in enumerate(zip(starts, outputs)):
            weight = hann.copy()
            if k == 0:
                weight[:HOP] = 1.0
            if k == len(starts) - 1:
                weight[HOP:] = 1.0
            acc[s:s + 512] += weight * out
        return acc

    board_sig = stitch([r.samples.astype(np.float64) for r in replies]) / args.gain
    ref_sig = stitch([reference.process(noisy_wire[s:s + 512]).samples.astype(np.float64) for s in starts]) / args.gain \
        if reference else None
    inner = slice(HOP, noisy_wire.size - HOP)  # drop half-covered edges from scoring
    quality = {"noisy": score(clean[inner], noisy_wire[inner] / args.gain),
               "board": score(clean[inner], board_sig[inner])}
    if ref_sig is not None:
        quality["reference"] = score(clean[inner], ref_sig[inner])
    hop_ms = HOP / SAMPLE_RATE_HZ * 1e3
    rtt = timing["round_trip"]
    summary = {
        "mode": "continuous", "record": args.record, "seconds": noisy_wire.size / SAMPLE_RATE_HZ,
        "snr_db": args.snr_db, "windows": len(starts), "input_clipped_samples": int(clipped),
        "quality": quality, "timing": timing, "hop_ms": hop_ms,
        "keeps_up_in_real_time": bool(rtt.get("p99_ms", float("inf")) < hop_ms),
        "stream_latency_ms": (512 / SAMPLE_RATE_HZ * 1e3 + rtt["mean_ms"]) if rtt.get("count") else None,
    }
    np.savez_compressed(out_dir / "host_continuous.npz", clean_mv=clean, noisy_int16=noisy_wire,
                        board_mv=board_sig, reference_mv=ref_sig if ref_sig is not None else np.zeros(0), gain=args.gain)
    write_csv(out_dir / "host_windows.csv", rows)
    if not args.no_plots:
        plot_continuous(out_dir, clean, noisy_wire / args.gain, board_sig, ref_sig)
        plot_latency(out_dir, rows)
    return summary


# ----------------------------------------------------------------------------- output

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k != "window", k))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_windows(out_dir, items, wire, board_out, ref_out, rows, gain) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.arange(512) / SAMPLE_RATE_HZ
    picks = np.argsort([r["prd_noisy"] for r in rows])[[len(rows) // 10, len(rows) // 2, (len(rows) * 9) // 10]]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3 * len(picks)), sharex=True)
    for ax, i in zip(np.atleast_1d(axes), picks):
        ax.plot(t, wire[i][0] / gain, color="#bbbbbb", lw=0.8, label="noisy input")
        ax.plot(t, items[i]["clean"], color="#2b7a3d", lw=1.0, label="clean reference")
        if ref_out is not None:
            ax.plot(t, ref_out[i] / gain, color="#e69f00", lw=1.0, ls="--", label="laptop float model")
        ax.plot(t, board_out[i] / gain, color="#1f4e9c", lw=1.2, label="ZCU104 output")
        ax.set_title(f"record {items[i]['record']} @ {items[i]['start']}: PRD noisy {rows[i]['prd_noisy']:.1f}% "
                     f"-> board {rows[i]['prd_board']:.1f}%", fontsize=10)
        ax.set_ylabel("mV")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "waveforms.png", dpi=130)
    plt.close(fig)
    plot_latency(out_dir, rows)


def plot_latency(out_dir, rows) -> None:
    import matplotlib.pyplot as plt

    good = [r for r in rows if r["attempts"] == 1]
    if not good:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([r["rtt_ms"] for r in good], bins=50, color="#1f4e9c", alpha=0.8, label="host round trip")
    ax.hist([r["board_ms"] for r in good], bins=50, color="#e69f00", alpha=0.8, label="on-board processing")
    ax.set_xlabel("ms")
    ax.set_ylabel("windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_hist.png", dpi=130)
    plt.close(fig)


def plot_continuous(out_dir, clean, noisy, board, ref) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    span = slice(0, min(clean.size, int(10 * SAMPLE_RATE_HZ)))
    t = np.arange(clean.size)[span] / SAMPLE_RATE_HZ
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, noisy[span], color="#bbbbbb", lw=0.8, label="noisy input")
    ax.plot(t, clean[span], color="#2b7a3d", lw=1.0, label="clean reference")
    if ref is not None:
        ax.plot(t, ref[span], color="#e69f00", lw=1.0, ls="--", label="laptop float model")
    ax.plot(t, board[span], color="#1f4e9c", lw=1.2, label="ZCU104 output")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.out_dir or ROOT / "results" / time.strftime(f"host_{args.mode}_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = make_reference(args)
    with open_transport(args.port, args.baud, args.timeout) as link:
        link.reset_input()
        client = BoardClient(link, retries=args.retries)
        info, ping_rtt = client.ping()
        print(f"Connected via {link.description}; ping {ping_rtt * 1e3:.1f} ms")
        print(f"Board: {json.dumps(info.get('backend', {}))}")
        if info.get("normalization") != args.normalization:
            print(f"WARNING: board normalization {info.get('normalization')} != host {args.normalization}")
        started = time.time()
        summary = (run_windows if args.mode == "windows" else run_continuous)(args, client, reference, out_dir)
        summary.update({"board_info": info, "link": link.description,
                        "baud": None if args.port.startswith("tcp://") else args.baud,
                        "gain_counts_per_mv": args.gain, "normalization": args.normalization,
                        "wall_seconds": time.time() - started, "client_stats": client.stats,
                        "reference_checkpoint": args.reference_checkpoint if reference else None})
        if not args.no_bye:
            client.bye()

    (out_dir / "host_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    q = summary["quality"]
    t = summary["timing"]
    print("\n== Result ==")
    for name in ("noisy", "board", "reference"):
        if q.get(name):
            print(f"  {name:<10} PRD {q[name]['prd']:6.2f}%  SNR {q[name]['snr_db']:5.2f} dB  MSE {q[name]['mse']:.4f}")
    agree = summary.get("board_vs_reference", {}).get("mean")
    if agree:
        print(f"  board vs laptop: RMSE {agree['rmse_mv'] * 1e3:.1f} uV ({agree['rmse_pct_of_span']:.2f}% of span), "
              f"corr {agree['corr']:.4f}")
    if t["round_trip"].get("count"):
        print(f"  round trip mean {t['round_trip']['mean_ms']:.1f} ms (p99 {t['round_trip']['p99_ms']:.1f}) | "
              f"board {t['board_total']['mean_ms']:.2f} ms | DPU {t['board_infer']['mean_ms']:.3f} ms"
              + (f" | wire (theory) {t['theoretical_wire_ms']:.1f} ms" if t["theoretical_wire_ms"] else ""))
    print(f"  client stats {client.stats}")
    print(f"Saved {out_dir}")
    return summary


if __name__ == "__main__":
    main()
