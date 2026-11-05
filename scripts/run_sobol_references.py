from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmc_sde_v4.sobol_references import run


parser = argparse.ArgumentParser(description="Compute independently scrambled Sobol reference values.")
parser.add_argument("--config", default=str(ROOT / "configs/sobol_reference.json"))
parser.add_argument("--output")
parser.add_argument("--device", choices=("cpu", "cuda"))
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
run(args.config, args.output, args.device, dry_run=args.dry_run)
