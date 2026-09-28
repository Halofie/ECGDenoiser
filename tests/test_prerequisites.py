from pathlib import Path

from scripts.check_fpga_prerequisites import main


def test_prerequisite_checker_reports_missing_checkpoint(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["check_fpga_prerequisites.py", "--checkpoint", str(tmp_path / "missing.pt")],
    )
    assert main() == 1
    assert "[MISSING] checkpoint" in capsys.readouterr().out
