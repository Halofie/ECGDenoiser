from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, validate_config
from src.data.dataset import ECGDenoisingDataset, load_noise_records
from src.data.fetch_dataset import load_clean_record
from src.models.quant_autoencoder import ECGDenoiseAutoencoder
from src.split.manifest import load_split_manifest
from src.training.metrics import mse, prd, snr_improvement


class VisualizerData:
    """Load the test split once and expose model predictions as JSON-ready data."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path):
        config_path = Path(config_path)
        root = config_path.resolve().parents[1]
        config = load_config(config_path)
        validate_config(config)

        clean_config = config["data"]["clean"]
        records = {
            name: load_clean_record(root / clean_config["path"] / name, channel=clean_config["channel"])
            for name in clean_config["records"]
        }
        manifest = load_split_manifest(root / config["splits"]["manifest"])
        noise_config = config["data"]["noise"]
        noise_records = load_noise_records(root / noise_config["path"], noise_config["types"])
        window = config["data"]["window"]
        seed = config["data"]["noise_injection"]["seed"]
        self.sample_rate_hz = float(config["data"]["signal"]["sample_rate_hz"])
        self.dataset = ECGDenoisingDataset(
            [records[name] for name in manifest["splits"]["test"]],
            noise_records,
            noise_config["weights"],
            snr_range_db=tuple(config["data"]["noise_injection"]["snr_range_db"]),
            window_length=window["length"],
            window_stride=window["stride"],
            seed=seed,
            deterministic=True,
        )

        self.model = ECGDenoiseAutoencoder()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        self.checkpoint_prd = checkpoint.get("validation_prd") if isinstance(checkpoint, dict) else None

    def sample(self, index: int) -> dict:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Sample index must be between 0 and {len(self.dataset) - 1}")
        noisy, clean, metadata = self.dataset[index]
        with torch.no_grad():
            denoised = self.model(noisy.unsqueeze(0)).squeeze(0).squeeze(0).numpy()
        noisy_values = noisy.squeeze(0).numpy()
        clean_values = clean.squeeze(0).numpy()
        residual = noisy_values - denoised
        return {
            "index": index,
            "total_samples": len(self.dataset),
            "sample_rate_hz": self.sample_rate_hz,
            "record": metadata["record"],
            "start": metadata["start"],
            "target_snr_db": metadata["target_snr_db"],
            "noisy": noisy_values.tolist(),
            "denoised": denoised.tolist(),
            "clean": clean_values.tolist(),
            "residual": residual.tolist(),
            "metrics": {
                "input_prd": prd(clean_values, noisy_values),
                "output_prd": prd(clean_values, denoised),
                "output_snr_db": snr_improvement(clean_values, denoised),
                "mse": mse(clean_values, denoised),
            },
        }


def make_handler(data: VisualizerData, web_root: Path, progress_path: Path):
    web_root = web_root.resolve()

    class VisualizerHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            request = urlparse(self.path)
            if request.path == "/api/health":
                self._send_json({"status": "ok", "samples": len(data.dataset)})
                return
            if request.path == "/api/training-progress":
                if not progress_path.is_file():
                    self._send_json({"status": "not_started", "completed_epochs": 0, "total_epochs": 60})
                    return
                try:
                    self._send_json(json.loads(progress_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError) as exc:
                    self._send_json({"status": "unavailable", "error": str(exc)}, status=503)
                return
            if request.path == "/api/sample":
                try:
                    index = int(parse_qs(request.query).get("index", ["0"])[0])
                    self._send_json(data.sample(index))
                except (IndexError, ValueError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return
            static_path = "/index.html" if request.path in ("/", "/index.html") else request.path
            requested = (web_root / static_path.lstrip("/")).resolve()
            if web_root not in requested.parents or not requested.is_file():
                self.send_error(404)
                return
            content_type = "text/html; charset=utf-8" if requested.suffix == ".html" else "text/css; charset=utf-8" if requested.suffix == ".css" else "application/javascript; charset=utf-8"
            body = requested.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    return VisualizerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ECG denoising waveform visualizer.")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--progress", default="runs/model_60_epochs_progress.json")
    args = parser.parse_args()

    data = VisualizerData(args.config, args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data, ROOT / "web", ROOT / args.progress))
    print(f"ECG visualizer running at http://{args.host}:{args.port}", flush=True)
    print(f"Loaded {len(data.dataset)} deterministic test windows", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()