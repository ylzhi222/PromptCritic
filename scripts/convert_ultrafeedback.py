"""
convert_ultrafeedback.py
------------------------
pip install datasets pandas tqdm
python scripts/convert_ultrafeedback.py --split train --out_csv data/ultrafeedback_promptcritic.csv
"""


import argparse, os, json, pandas as pd
from tqdm import tqdm
from datasets import load_dataset


# ─── 把 0-10 分映射成 0/1/2 ──────────────────────────────
def score_to_label(score: float) -> int:
    if score < 4:        # 0 – 3.9   → low
        return 0
    if score < 7.5:      # 4 – 7.4   → medium
        return 1
    return 2             # 7.5 – 10  → high


def convert(split: str, out_csv: str, take_all: bool = False):
    ds = load_dataset("openbmb/UltraFeedback", split=split)
    rows = []

    for item in tqdm(ds, desc=f"processing {split}"):
        prompt = item.get("instruction", "")
        if not prompt:
            continue

        # 新版字段叫 completions
        for comp in item.get("completions", []):
            resp_txt  = comp.get("response", "")
            score     = comp.get("overall_score")
            if resp_txt and score is not None:
                label = score_to_label(float(score))
                rows.append({"prompt": prompt,
                             "response": resp_txt,
                             "label": label})
            if not take_all:
                break   # 只取第一条回答

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅  Saved {len(df)} rows -> {out_csv}")


# ─── CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    help="train / validation / test")
    ap.add_argument("--out_csv", default="data/ultra_promptcritic.csv")
    ap.add_argument("--take_all", action="store_true",
                    help="keep every completion for each prompt")
    args = ap.parse_args()
    convert(args.split, args.out_csv, args.take_all)
