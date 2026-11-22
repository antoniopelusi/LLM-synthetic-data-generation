from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def hf(repo, out):
    path = DATA / out
    if path.exists():
        print(f"[skip] {out} already exists")
        return
    from datasets import load_dataset
    print(f"[hf] {repo} -> {out}")
    ds = load_dataset(repo, split="train")
    print(f"     {len(ds)} rows, columns: {ds.column_names}")
    ds.to_parquet(str(path))


def bank():
    path = DATA / "bank.csv"
    if path.exists():
        print("[skip] bank.csv already exists")
        return
    print("[kaggle] janiobachmann/bank-marketing-dataset")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("janiobachmann/bank-marketing-dataset",
                                   path=str(DATA), unzip=True)
    except BaseException as e:
        print(f"[!] Kaggle download failed ({type(e).__name__}: {e}).\n"
              "    Set your credentials in ~/.kaggle/kaggle.json, or download bank.csv manually from\n"
              "    https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset\n"
              f"    and place it in {DATA}.")


if __name__ == "__main__":
    hf("pointe77/credit-card-transaction", "credit.parquet")
    hf("scikit-learn/adult-census-income", "adult.parquet")
    bank()
