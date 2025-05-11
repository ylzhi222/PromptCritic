"""
Dataset & DataLoader builder for PromptCritic
Usage (示例):
    python scripts/dataset_builder.py \
           --csv_path data/sharegpt_labeled.csv \
           --model_name bert-base-uncased \
           --max_len 256
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------------- Dataset 定义 --------------------------- #
class PromptResponseDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"[PROMPT] {row['prompt']} [RESPONSE] {row['response']}"
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        label = int(row["label"]) if row["label"] == row["label"] else -1  # 未标注设为 -1
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item

# --------------------------- 构建 DataLoader --------------------------- #
def build_loaders(
    csv_path: str,
    model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    max_len: int = 256,
    val_size: float = 0.2,
    num_workers: int = 0,
):
    df = pd.read_csv(csv_path)
    # 仅保留已标注样本
    df_labeled = df[df["label"].notna()].copy()

    train_df, val_df = train_test_split(
        df_labeled, test_size=val_size, stratify=df_labeled["label"], random_state=42
    )

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_ds = PromptResponseDataset(train_df, tokenizer, max_len)
    val_ds = PromptResponseDataset(val_df, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer


# --------------------------- CLI --------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    tl, vl, _ = build_loaders(
        csv_path=args.csv_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # 简单 sanity check
    print(f"✅ Train batches: {len(tl)},  Val batches: {len(vl)}")
    for batch in tl:
        print({k: v.shape for k, v in batch.items()})
        break
