import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--members")
parser.add_argument("--nonmembers")
parser.add_argument("--synthetic")
parser.add_argument("--features", choices=["all", "num", "cat"], default="all")
parser.add_argument("--max-samples", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir")
args = parser.parse_args()

print("===========================| DcrMIA started...\n")

ROOT = Path(__file__).resolve().parent.parent
load = lambda p: (lambda r: r["records"] if isinstance(r, dict) else r)(json.loads(Path(p).read_text(encoding="utf-8")))

config_path = Path(args.config)
if not config_path.is_absolute() and (ROOT / "configs" / config_path).exists():
    config_path = ROOT / "configs" / config_path
config = json.loads(config_path.read_text(encoding="utf-8"))
stem = config_path.stem
dataset_dir = (ROOT / "generation" / config["dataset_dir"]).resolve()
output_dir = (ROOT / "generation" / config["output_dir"]).resolve()
out_dir = Path(args.out_dir or ROOT / "outputs" / "dcr-mia") / stem

icl_path = None
if config["strategy"].startswith("few_shot_icl"):
    n_icl = int(config["strategy"].replace("few_shot_icl_", "").replace("r", ""))
    icl_path = dataset_dir / next(
        e["file"] for e in config["datasets"] if isinstance(e["sample"], int) and e["sample"] == n_icl
    )

if args.members:
    members_path = Path(args.members)
elif config.get("dataset"):
    members_path = dataset_dir / Path(config["dataset"]).name
elif icl_path:
    members_path = icl_path
else:
    sys.exit(f"  error: no member records for strategy {config['strategy']}, pass --members")
synthetic_path = Path(args.synthetic or output_dir / f"{stem}_{config['seed']}.json")

members = load(members_path)
synthetic = load(synthetic_path)
if args.nonmembers:
    nonmembers = load(args.nonmembers)
else:
    seen = {json.dumps(r, sort_keys=True) for r in members}
    nonmembers = [r for r in load(dataset_dir / config["original_dataset"]) if json.dumps(r, sort_keys=True) not in seen]

num_cols = config["num_cols"]
cat_cols = config["cat_cols"]
all_cols = num_cols + cat_cols
train_df = pd.DataFrame(members)[all_cols].dropna()
synt_df = pd.DataFrame(synthetic)[all_cols].dropna()
test_df = pd.DataFrame(nonmembers)[all_cols].dropna()

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_df[cat_cols])
arrays = [np.hstack([df[num_cols].values, enc.transform(df[cat_cols])]) for df in (train_df, synt_df, test_df)]
scaler = StandardScaler().fit(arrays[0])
train_x, synt_x, test_x = [scaler.transform(a) for a in arrays]

n = min(len(train_x), len(test_x), args.max_samples or len(train_x))
rng = np.random.default_rng(args.seed)
idx_m = rng.choice(len(train_x), n, replace=False)
idx_nm = rng.choice(len(test_x), n, replace=False)
members_x, nonmembers_x = train_x[idx_m], test_x[idx_nm]

n_num = len(num_cols)
cols = {"all": slice(None), "num": slice(0, n_num), "cat": slice(n_num, None)}[args.features]
members_x, nonmembers_x, synt_x = members_x[:, cols], nonmembers_x[:, cols], synt_x[:, cols]

nn = NearestNeighbors(n_neighbors=1).fit(synt_x)
dist_m = nn.kneighbors(members_x)[0].flatten()
dist_nm = nn.kneighbors(nonmembers_x)[0].flatten()
scores = np.concatenate([-dist_m, -dist_nm])
labels = np.concatenate([np.ones(n), np.zeros(n)])

fpr, tpr, thresholds = roc_curve(labels, scores)
auc = round(float(roc_auc_score(labels, scores)), 4)
acc = round(float(np.mean((scores >= thresholds[np.argmax(tpr - fpr)]).astype(int) == labels)), 4)
tpr_at = lambda t: round(float(tpr[min(max(0, int(np.searchsorted(fpr, t, side="right")) - 1), len(tpr) - 1)]) * 100, 4)
metrics = {"attack": "dcr-mia", "config": stem, "n_members": n, "n_nonmembers": n, "auc": auc, "accuracy": acc, "tpr_1": tpr_at(0.01), "tpr_01": tpr_at(0.001)}

out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "group": ["member"] * n + ["non_member"] * n,
        "label": labels.astype(int),
        "sample_id": np.concatenate([np.arange(n), np.arange(n)]),
        "dcr": np.concatenate([dist_m, dist_nm]),
        "score": scores,
    }
).to_csv(out_dir / "results.csv", index=False)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

for k, v in metrics.items():
    print(f"  {k:<22}: {v}")
print(f"  {'saved':<22}: {out_dir}")

print("\n===========================| DcrMIA completed.")
