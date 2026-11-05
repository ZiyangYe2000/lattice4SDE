from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qmc_sde_v4.plotting import plot_tables


parser = argparse.ArgumentParser(description="Plot RMSE figures from rmse_summary.csv only.")
parser.add_argument("--rmse", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
print(json.dumps(plot_tables(args.rmse, args.output), indent=2))
