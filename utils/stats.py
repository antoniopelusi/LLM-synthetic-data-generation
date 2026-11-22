import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

VOCAB_MAX = 20
TOP_K = 20

DATASETS = {
    "credit": dict(file="credit.parquet",
                   force_cat=["category", "gender", "state", "job", "is_fraud"]),
    "adult": dict(file="adult.parquet", force_cat=[]),
    "bank": dict(file="bank.csv", force_cat=[]),
}

def load(spec):
    p = DATA / spec["file"]
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def describe_numeric(s):
    s = pd.to_numeric(s, errors="coerce")
    x = s.dropna()
    q = x.quantile([.01, .05, .25, .5, .75, .95, .99])
    return {
        "type": "numeric", "count": len(x), "missing": int(s.isna().sum()),
        "n_unique": int(x.nunique()), "min": x.min(), "q01": q[.01], "q05": q[.05],
        "q25": q[.25], "median": q[.5], "q75": q[.75], "q95": q[.95], "q99": q[.99],
        "max": x.max(), "mean": x.mean(), "std": x.std(),
        "skew": x.skew(), "excess_kurtosis": x.kurt(), "share_zero": float((x == 0).mean()),
    }


def describe_categorical(s):
    n = len(s)
    st = s.astype("string")
    miss = {"nan": int(s.isna().sum()),
            "empty_string": int((st == "").sum()),
            "question_mark": int((st == "?").sum())}
    vc = st.value_counts(dropna=True)
    share = vc / n
    p = vc / vc.sum()
    ent = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    out = {"type": "categorical", "n_unique": int(len(vc)), "missing_like": miss,
           "top1_share": float(share.iloc[0]) if len(share) else None,
           "normalized_entropy": ent}
    keep = share if len(vc) < VOCAB_MAX else share.head(TOP_K)
    out["frequencies" if len(vc) < VOCAB_MAX else f"top{TOP_K}"] = {str(k): float(v) for k, v in keep.items()}
    return out


def describe(df, spec):
    stats = {}
    for c in df.columns:
        s = df[c]
        if c in spec["force_cat"] or s.dtype == object or str(s.dtype) in ("string", "category", "bool"):
            stats[c] = describe_categorical(s)
        elif pd.api.types.is_numeric_dtype(s):
            stats[c] = describe_numeric(s)
        else:
            stats[c] = describe_categorical(s)
    return stats


def credit_extras(df):
    e = {}
    z_raw = df["zip"]
    z = pd.to_numeric(z_raw, errors="coerce")
    mraw = df["merch_zipcode"]
    mz = pd.to_numeric(mraw.replace("", np.nan) if mraw.dtype == object else mraw, errors="coerce")
    e["zip_dtype"] = str(z_raw.dtype)
    e["merch_zipcode_dtype"] = str(mraw.dtype)
    if z_raw.dtype == object:
        zs = z_raw.astype("string")
        e["zip_share_len_lt_5"] = float((zs.str.len() < 5).mean())
        e["zip_share_leading_zero"] = float(zs.str.startswith("0").mean())
    e["zip_min"], e["zip_max"] = z.min(), z.max()
    e["zip_missing"] = int(z.isna().sum())
    e["merch_zipcode_present_share"] = float(mz.notna().mean())
    e["merch_zipcode_min"], e["merch_zipcode_max"] = mz.min(), mz.max()
    dz = (mz - z).abs().dropna()
    e["merch_zip_minus_zip_abs_quantiles"] = dz.quantile([.5, .9, .99, 1.0]).to_dict()
    e["merch_zip_within_1000_share"] = float((dz <= 1000).mean())
    e["merch_zip_within_20000_share"] = float((dz <= 20000).mean())
    dlat = (df["merch_lat"] - df["lat"]).abs()
    dlon = (df["merch_long"] - df["long"]).abs()
    e["abs_merch_lat_minus_lat_quantiles"] = dlat.quantile([.5, .9, .99, 1.0]).to_dict()
    e["abs_merch_long_minus_long_quantiles"] = dlon.quantile([.5, .9, .99, 1.0]).to_dict()
    e["share_dlat_lt_1.0"] = float((dlat < 1.0).mean())
    e["share_dlong_lt_0.3"] = float((dlon < 0.3).mean())
    e["corr_lat_merch_lat"] = float(df["lat"].corr(df["merch_lat"]))
    e["corr_long_merch_long"] = float(df["long"].corr(df["merch_long"]))
    e["corr_zip_long"] = float(z.corr(df["long"]))
    dob = pd.to_datetime(df["dob"], errors="coerce")
    tx = pd.to_datetime(df["unix_time"], unit="s")
    age = (tx - dob).dt.days / 365.25
    e["dob_year_min"], e["dob_year_max"] = int(dob.dt.year.min()), int(dob.dt.year.max())
    e["age_at_tx_min"], e["age_at_tx_median"], e["age_at_tx_max"] = age.min(), age.median(), age.max()
    e["share_age_lt_18"] = float((age < 18).mean())
    e["tx_date_min"], e["tx_date_max"] = str(tx.min()), str(tx.max())
    tn = df["trans_num"].astype("string")
    e["trans_num_len_set"] = sorted({int(v) for v in tn.str.len().unique()})
    e["trans_num_all_lower_hex"] = bool(tn.str.fullmatch(r"[0-9a-f]{32}").all())
    e["trans_num_unique_share"] = float(tn.nunique() / len(tn))
    m = df["merchant"].astype("string")
    e["merchant_share_prefix_fraud_"] = float(m.str.startswith("fraud_").mean())
    suffix = m.str.extract(r"(LLC|PLC|Ltd|Inc|Group)$")[0]
    e["merchant_suffix_shares"] = suffix.fillna("other").value_counts(normalize=True).to_dict()
    e["merchant_share_with_comma_and"] = float(m.str.contains(r", .* and ").mean())
    e["merchant_share_hyphen"] = float(m.str.contains("-").mean())
    e["fraud_rate_by_category"] = df.groupby("category")["is_fraud"].mean().round(5).to_dict()
    return {k: (py(v) if not isinstance(v, (dict, list)) else v) for k, v in e.items()}


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.6g}"


def md_dataset(name, df, stats):
    num = {c: s for c, s in stats.items() if s["type"] == "numeric"}
    cat = {c: s for c, s in stats.items() if s["type"] == "categorical"}
    L = [f"## {name} ({len(df):,} rows, {df.shape[1]} columns)\n", "### Numeric\n",
         "| Feature | missing | min | q25 | median | q75 | max | mean | std | skew | exc. kurt |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in num.items():
        L.append(f"| {c} | {s['missing']} | " + " | ".join(fmt(s[k]) for k in
                 ["min", "q25", "median", "q75", "max", "mean", "std", "skew", "excess_kurtosis"]) + " |")
    L += ["", "### Categorical\n", "| Feature | n unique | top1 | norm. entropy | missing (nan/''/?) | values (share) |",
          "|---|---|---|---|---|---|"]
    for c, s in cat.items():
        fr = s.get("frequencies") or s.get(f"top{TOP_K}")
        shown = list(fr.items())[:10] if s["n_unique"] >= VOCAB_MAX else list(fr.items())
        vals = ", ".join(f"{k}: {v:.2%}" for k, v in shown)
        mm = s["missing_like"]
        L.append(f"| {c} | {s['n_unique']} | {fmt(s['top1_share'])} | {s['normalized_entropy']:.3f} | "
                 f"{mm['nan']}/{mm['empty_string']}/{mm['question_mark']} | {vals} |")
    return "\n".join(L) + "\n"


def main():
    report = ["# Empirical dataset statistics\n"]
    for name, spec in DATASETS.items():
        df = load(spec)
        if df is None:
            print(f"[!] {spec['file']} not found in {DATA}: skipping {name}")
            report.append(f"## {name}\n\nFile `{spec['file']}` not found.\n")
            continue
        print(f"[{name}] {len(df):,} rows")
        stats = describe(df, spec)
        payload = {"n_rows": len(df), "columns": list(df.columns), "stats": stats}
        if name == "credit":
            extras = credit_extras(df)
            payload["credit_extras"] = extras
        (OUT / f"stats_{name}.json").write_text(json.dumps(payload, indent=2, default=py))
        report.append(md_dataset(name, df, stats))
        if name == "credit":
            report.append("### Credit: additional statistics\n\n```json\n"
                          + json.dumps(extras, indent=2, default=py) + "\n```\n")
    (OUT / "stats.md").write_text("\n".join(report))
    print(f"Done. Output in {OUT}")


if __name__ == "__main__":
    main()
