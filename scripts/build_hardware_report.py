"""Turn board-day result folders into reports/hardware_results.md (+ figures).

    python scripts/build_hardware_report.py \
        --session results/session_20261101_101500 \
        --server results/uart_20261101_103000 \
        --host results/host_windows_20261101_103010 \
        --host-continuous results/host_continuous_20261101_104000 \
        --golden-report reports/dpu_validation/dpu_validation_20261101T101900Z.json \
        --quant-eval vitis_ai_deployment/artifacts/quantize_result_noisy_minmax/quant_eval.json

Every input is optional; sections without data are reported as "not measured"
rather than silently omitted. Runs whose backend is not the DPU (laptop
rehearsals) are stamped SIMULATED at the top of the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NA = "not measured"


def load(path: Path | None, name: str) -> dict | None:
    if path is None:
        return None
    target = path / name if path.is_dir() else path
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else None


def f(value, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{unit}"


def lat_row(label: str, stats: dict | None, note: str = "") -> str:
    if not stats or not stats.get("count"):
        return f"| {label} | — | — | — | — | — | {note or NA} |"
    return (f"| {label} | {f(stats['mean_ms'], 3)} | {f(stats['p50_ms'], 3)} | {f(stats['p95_ms'], 3)} | "
            f"{f(stats['p99_ms'], 3)} | {f(stats['max_ms'], 3)} | {note} n={stats['count']} |")


def section_setup(lines, session, server, host) -> None:
    env = (session or {}).get("env", {})
    backend = (session or {}).get("backend") or (host or {}).get("board_info", {}).get("backend") or {}
    lines += ["## 1. Setup", "", "| Item | Value |", "|---|---|"]
    rows = [
        ("Backend", backend.get("backend")),
        ("Board host / platform", f"{env.get('hostname', '—')} / {env.get('platform', '—')}"),
        ("Python / numpy / VART", f"{env.get('python', '—')} / {env.get('numpy', '—')} / {env.get('vart_version', '—')}"),
        ("xmodel", f"{backend.get('xmodel', backend.get('checkpoint', '—'))} (md5 `{backend.get('xmodel_md5', backend.get('checkpoint_md5', '—'))}`)"),
        ("DPU subgraphs", backend.get("dpu_subgraphs")),
        ("DPU input tensor", backend.get("input")),
        ("DPU output tensor", backend.get("output")),
        ("Normalization", (session or host or {}).get("normalization")),
        ("Host link", (host or {}).get("link") or (server or {}).get("link")),
        ("Baud", (host or {}).get("baud")),
        ("Wire format", f"int16, {(host or {}).get('gain_counts_per_mv', '—')} counts/mV"),
    ]
    lines += [f"| {k} | {v if v not in (None, '') else '—'} |" for k, v in rows]
    lines.append("")


def section_correctness(lines, golden_report, host, quant_eval) -> None:
    lines += ["## 2. Correctness", ""]
    if golden_report:
        s = golden_report["summary"]
        lines.append(f"**Golden vectors (DPU vs float model, normalized domain):** {s['num_ok']}/{s['num_vectors_compared']} OK, "
                     f"{s['num_high_error']} high-error, {s['num_shape_mismatch']} shape mismatch, "
                     f"{s['num_non_finite']} non-finite; mean MSE {f(s.get('mean_mse'), 6)}.")
    else:
        lines.append(f"**Golden vectors:** {NA} (run compare_golden_results.py on the session's golden_outputs/).")
    agree = ((host or {}).get("board_vs_reference") or {}).get("mean")
    if agree:
        lines.append("")
        lines.append(f"**Full UART pipeline vs laptop float model (same int16 input, {host['num_windows']} windows):** "
                     f"RMSE {f(agree['rmse_mv'] * 1e3, 1)} µV = {f(agree['rmse_pct_of_span'], 2)}% of signal span "
                     f"(worst window {f(host['board_vs_reference']['worst_rmse_pct_of_span'], 2)}%), "
                     f"correlation {f(agree['corr'], 4)}.")
    if quant_eval:
        lines.append("")
        lines.append(f"**Quantization (Vitis AI fake-quant, {quant_eval['num_windows']} windows):** float PRD "
                     f"{f(quant_eval['float']['prd_percent'])}% → INT8 PRD {f(quant_eval['int8_fake_quant']['prd_percent'])}% "
                     f"({quant_eval['int8_minus_float_prd_points']:+.2f} points).")
    lines.append("")


def section_quality(lines, host, continuous) -> None:
    lines += ["## 3. Denoising quality on the board", ""]
    if not host and not continuous:
        lines += [NA, ""]
        return
    lines += ["PRD/SNR/MSE in the clean window's [-1, 1] scale (same convention as `reports/results_and_analysis.md`).",
              "", "| Run | Signal | PRD % | SNR dB | MSE |", "|---|---|---:|---:|---:|"]
    for label, run in (("held-out windows", host), ("continuous stream", continuous)):
        if not run:
            continue
        if run is continuous:
            label = f"record {run['record']}, {f(run['seconds'], 0)} s @ {run['snr_db']} dB"
        for name, title in (("noisy", "noisy input"), ("board", "**ZCU104 output**"), ("reference", "laptop float model")):
            q = run["quality"].get(name)
            if q:
                lines.append(f"| {label} | {title} | {f(q['prd'])} | {f(q['snr_db'])} | {f(q['mse'], 4)} |")
    lines.append("")


def section_latency(lines, session, server, host, continuous) -> None:
    lines += ["## 4. Latency", "", "| Stage | mean ms | p50 | p95 | p99 | max | notes |", "|---|---:|---:|---:|---:|---:|---|"]
    lat = (session or {}).get("latency", {})
    lines.append(lat_row("DPU only (execute_async+wait)", lat.get("infer"), "board_session, single thread"))
    lines.append(lat_row("Pre-processing (normalize+quantize)", lat.get("pipeline_pre"), "board_session"))
    lines.append(lat_row("Post-processing (dequantize+restore)", lat.get("pipeline_post"), "board_session"))
    lines.append(lat_row("Full on-board pipeline", lat.get("pipeline_total"), "board_session"))
    srv = (server or {}).get("latency", {})
    lines.append(lat_row("On-board per request (UART service)", srv.get("board_s"), "uart_server"))
    timing = (host or {}).get("timing", {})
    lines.append(lat_row("Host round trip (UART)", timing.get("round_trip"), "host, incl. wire time"))
    lines.append(lat_row("Link + host overhead", timing.get("link_and_host"), "round trip − board time"))
    lines.append("")
    if timing.get("theoretical_wire_ms"):
        lines.append(f"Theoretical wire time per round trip at {host['baud']} baud: "
                     f"{f(timing['theoretical_wire_ms'], 1)} ms (1038 B request + 1058 B response, 10 bits/byte).")
    if continuous:
        lines.append(f"Real-time stream: hop {f(continuous['hop_ms'], 1)} ms; p99 round trip "
                     f"{f(continuous['timing']['round_trip'].get('p99_ms'), 1)} ms → keeps up: "
                     f"**{'yes' if continuous['keeps_up_in_real_time'] else 'NO'}**; end-to-end stream latency "
                     f"(window fill + round trip) ≈ {f(continuous.get('stream_latency_ms'), 0)} ms.")
    thr = (session or {}).get("throughput", [])
    if thr:
        lines += ["", "| Threads/runners | windows/s | per-call mean ms | per-call p99 ms |", "|---:|---:|---:|---:|"]
        for t in thr:
            lines.append(f"| {t['threads']} | {f(t['windows_per_s'], 1)} | {f(t['per_call'].get('mean_ms'), 3)} | "
                         f"{f(t['per_call'].get('p99_ms'), 3)} |")
    lines.append("")


def section_power(lines, session, server) -> None:
    lines += ["## 5. Power and energy", ""]
    power = (session or {}).get("power") or {}
    if not power.get("available"):
        lines += [f"{NA} — no hwmon power channels were found on the board image. "
                  "Use an external meter on the 12 V input (see runbook).", ""]
        return
    phases = power["phases"]
    order = [p for p in ("idle", "latency_single") if p in phases] + sorted(p for p in phases if p.startswith("saturated"))
    channels = [c["name"] for c in power["channels"]]
    lines += [f"Channels summed into total: {', '.join('`' + k + '`' for k in power['total_channels'])}", "",
              "| Phase | total W (mean ± std) | " + " | ".join(channels) + " |",
              "|---|---:|" + "---:|" * len(channels)]
    for phase in order:
        row = phases[phase]
        total = row.get("total") or {}
        cells = [f(row.get(c, {}).get("mean_w") if row.get(c) else None, 3) for c in channels]
        lines.append(f"| {phase} | {f(total.get('mean_w'), 3)} ± {f(total.get('std_w'), 3)} | " + " | ".join(cells) + " |")
    srv_phases = ((server or {}).get("power") or {}).get("phases", {})
    if "uart_session" in srv_phases:
        total = srv_phases["uart_session"].get("total") or {}
        lines.append(f"| uart_session (live UART service) | {f(total.get('mean_w'), 3)} ± {f(total.get('std_w'), 3)} | "
                     + " | ".join("" for _ in channels) + " |")
    energy = (session or {}).get("energy", {})
    if energy.get("per_threads"):
        lines += ["", f"Idle total: {f(energy['idle_total_w'], 3)} W.", "",
                  "| Threads | windows/s | active W | dynamic W | mJ/window (total) | mJ/window (dynamic) |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for e in energy["per_threads"]:
            lines.append(f"| {e['threads']} | {f(e['windows_per_s'], 1)} | {f(e['active_total_w'], 3)} | "
                         f"{f(e['dynamic_w'], 3)} | {f(e['energy_per_window_total_mj'], 3)} | "
                         f"{f(e['energy_per_window_dynamic_mj'], 3)} |")
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--session", type=Path, help="board_session.py output folder (extracted).")
    parser.add_argument("--server", type=Path, help="uart_server.py output folder.")
    parser.add_argument("--host", type=Path, help="uart_host_client.py --mode windows output folder.")
    parser.add_argument("--host-continuous", type=Path, help="uart_host_client.py --mode continuous output folder.")
    parser.add_argument("--golden-report", type=Path, help="compare_golden_results.py JSON report.")
    parser.add_argument("--quant-eval", type=Path, help="quant_eval.json from quantize.py --evaluate.")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "hardware_results.md")
    parser.add_argument("--title", default="ECG Denoiser on ZCU104 — Hardware Results")
    args = parser.parse_args()

    session = load(args.session, "session.json")
    server = load(args.server, "server_summary.json")
    host = load(args.host, "host_summary.json")
    continuous = load(args.host_continuous, "host_summary.json")
    golden = load(args.golden_report, "")
    quant_eval = load(args.quant_eval, "")

    backends = {b for b in [((session or {}).get("backend") or {}).get("backend"),
                            ((host or {}).get("board_info") or {}).get("backend", {}).get("backend")] if b}
    simulated = bool(backends - {"dpu"})
    lines = [f"# {args.title}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/build_hardware_report.py`.", ""]
    if simulated:
        lines += [f"> **SIMULATED RUN** — backend(s) {sorted(backends)}, not the ZCU104 DPU. Numbers below exercise the "
                  "tooling only and must not be quoted as hardware results.", ""]
    section_setup(lines, session, server, host)
    section_correctness(lines, golden, host, quant_eval)
    section_quality(lines, host, continuous)
    section_latency(lines, session, server, host, continuous)
    section_power(lines, session, server)

    figures = []
    fig_dir = args.out.parent / "figures"
    for folder, names, prefix in ((args.host, ("waveforms.png", "latency_hist.png"), "hw_windows"),
                                  (args.host_continuous, ("continuous.png", "latency_hist.png"), "hw_continuous")):
        for name in names:
            if folder and (folder / name).is_file():
                fig_dir.mkdir(parents=True, exist_ok=True)
                target = fig_dir / f"{prefix}_{name}"
                shutil.copy2(folder / name, target)
                figures.append(target)
    lines += ["## 6. Figures", ""]
    lines += [f"![{p.stem}](figures/{p.name})" for p in figures] or [NA]
    lines += ["", "## 7. Sources", ""]
    for label, path in (("board session", args.session), ("UART server", args.server), ("host windows", args.host),
                        ("host continuous", args.host_continuous), ("golden report", args.golden_report),
                        ("quantization check", args.quant_eval)):
        lines.append(f"- {label}: `{path}`" if path else f"- {label}: {NA}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}" + (" (SIMULATED)" if simulated else ""))


if __name__ == "__main__":
    main()
