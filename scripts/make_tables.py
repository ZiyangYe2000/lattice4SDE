from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmc_sde_v4.analysis import build_tables


parser = argparse.ArgumentParser(description="Build the RMSE and slope tables used in the paper.")
parser.add_argument("--config", default=str(ROOT / "configs/production.json"))
parser.add_argument("--formal", required=True)
parser.add_argument("--reference-summary", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--allow-partial", action="store_true")
args = parser.parse_args()
report = build_tables(args.config, args.formal, args.reference_summary, args.output,
                      allow_partial=args.allow_partial)
print(json.dumps(report, indent=2))
