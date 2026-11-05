from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmc_sde_v4.formal import run


parser = argparse.ArgumentParser(description="Run equal-work MC and randomized lattice experiments.")
parser.add_argument("--config", default=str(ROOT / "configs/production.json"))
parser.add_argument("--output")
parser.add_argument("--device", choices=("cpu", "cuda"))
parser.add_argument("--max-tasks", type=int)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
run(args.config, args.output, args.device, args.max_tasks, dry_run=args.dry_run)
