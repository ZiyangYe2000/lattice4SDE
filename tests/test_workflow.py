from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_workflow(tmp_path):
    config = ROOT / "configs/smoke.json"
    sobol_config = ROOT / "configs/sobol_reference_smoke.json"
    reference = tmp_path / "reference"
    formal = tmp_path / "formal"
    analysis = tmp_path / "analysis"
    subprocess.run([sys.executable, str(ROOT / "scripts/run_sobol_references.py"), "--config", str(sobol_config),
                    "--output", str(reference), "--device", "cpu"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_experiments.py"), "--config", str(config),
                    "--output", str(formal), "--device", "cpu"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tables.py"), "--config", str(config),
                    "--formal", str(formal / "raw/formal.jsonl"), "--reference-summary",
                    str(reference / "tables/reference_summary.csv"), "--output", str(analysis)], check=True)
    assert (analysis / "analysis_manifest.json").exists()


def test_sobol_reference_smoke(tmp_path):
    config = ROOT / "configs/sobol_reference_smoke.json"
    output = tmp_path / "sobol_reference"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_sobol_references.py"),
        "--config", str(config), "--output", str(output), "--device", "cpu",
    ], check=True)
    assert (output / "tables/reference_summary.csv").exists()
    assert (output / "raw/sobol_components.jsonl").exists()
