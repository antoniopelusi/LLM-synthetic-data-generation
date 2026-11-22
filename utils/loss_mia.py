import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--members")
parser.add_argument("--nonmembers")
parser.add_argument("--adapter")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-samples", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir")
args = parser.parse_args()

print("===========================| LossMIA started...\n")

ROOT = Path(__file__).resolve().parent.parent
load = lambda p: (lambda r: r["records"] if isinstance(r, dict) else r)(json.loads(Path(p).read_text(encoding="utf-8")))

config_path = Path(args.config)
if not config_path.is_absolute() and (ROOT / "configs" / config_path).exists():
    config_path = ROOT / "configs" / config_path
config = json.loads(config_path.read_text(encoding="utf-8"))
stem = config_path.stem
dataset_dir = (ROOT / "generation" / config["dataset_dir"]).resolve()
output_dir = (ROOT / "generation" / config["output_dir"]).resolve()
out_dir = Path(args.out_dir or ROOT / "outputs" / "loss-mia") / stem

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
adapter_path = Path(args.adapter or output_dir / f"{stem}_model")
if config["strategy"] == "lora" and not adapter_path.exists():
    sys.exit(f"  error: adapter not found: {adapter_path}")

members = load(members_path)
if args.nonmembers:
    nonmembers = load(args.nonmembers)
else:
    seen = {json.dumps(r, sort_keys=True) for r in members}
    nonmembers = [r for r in load(dataset_dir / config["original_dataset"]) if json.dumps(r, sort_keys=True) not in seen]

all_cols = config["num_cols"] + config["cat_cols"]
members = [r for r, k in zip(members, pd.DataFrame(members)[all_cols].notna().all(axis=1)) if k]
nonmembers = [r for r, k in zip(nonmembers, pd.DataFrame(nonmembers)[all_cols].notna().all(axis=1)) if k]

n = min(len(members), len(nonmembers), args.max_samples or len(members))
rng = np.random.default_rng(args.seed)
idx_m = rng.choice(len(members), n, replace=False)
idx_nm = rng.choice(len(nonmembers), n, replace=False)
groups = [("member", [members[i] for i in idx_m], 1), ("non_member", [nonmembers[i] for i in idx_nm], 0)]

system_prompt = config["prompt"]["system"]
user_prompt = config["prompt"]["user"] + "\n"
assistant_prefix = config["prompt"]["assistant"]
if icl_path:
    icl_text = icl_path.read_text(encoding="utf-8").strip()
    user_prompt += (
        "\n# EXAMPLES\nThe following are real records shown only to illustrate the expected format and value ranges.\n"
        + "\n".join(icl_text.splitlines()[1:-1]).strip()
        + "\n"
    )

model_name = config["model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device if device == "cuda" else None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
if adapter_path.exists():
    model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    tokenize=False,
    add_generation_prompt=True,
) + assistant_prefix
prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

rows = []
for group, records, label in groups:
    for b in tqdm(range(0, n, args.batch_size), desc=f"  loss-mia {group:<10}", leave=False):
        batch = records[b : b + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prefix + json.dumps(r, ensure_ascii=False) + "\n```"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for r in batch
        ]
        enc = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_labels = labels[:, 1:]
            valid = shift_labels != -100
            token_losses = torch.zeros(shift_labels.shape, dtype=torch.float32, device=shift_labels.device)
            token_losses[valid] = torch.nn.functional.cross_entropy(
                logits[:, :-1, :][valid].float(), shift_labels[valid], reduction="none"
            )
            n_tokens = valid.sum(dim=1)
            losses = (token_losses.sum(dim=1) / n_tokens.clamp(min=1)).cpu().tolist()

        for i, (loss, n_tok) in enumerate(zip(losses, n_tokens.cpu().tolist())):
            rows.append(
                {"group": group, "label": label, "sample_id": b + i, "n_tokens": n_tok, "target_loss": loss, "score": -loss}
            )

df = pd.DataFrame(rows)
scores = df["score"].values
labels = df["label"].values

fpr, tpr, thresholds = roc_curve(labels, scores)
auc = round(float(roc_auc_score(labels, scores)), 4)
acc = round(float(np.mean((scores >= thresholds[np.argmax(tpr - fpr)]).astype(int) == labels)), 4)
tpr_at = lambda t: round(float(tpr[min(max(0, int(np.searchsorted(fpr, t, side="right")) - 1), len(tpr) - 1)]) * 100, 4)
metrics = {"attack": "loss-mia", "config": stem, "n_members": n, "n_nonmembers": n, "auc": auc, "accuracy": acc, "tpr_1": tpr_at(0.01), "tpr_01": tpr_at(0.001)}

out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "results.csv", index=False)
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

for k, v in metrics.items():
    print(f"  {k:<22}: {v}")
print(f"  {'saved':<22}: {out_dir}")

print("\n===========================| LossMIA completed.")
