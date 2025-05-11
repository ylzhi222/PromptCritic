"""
Evaluate PromptCritic model on labeled CSV
=========================================

示例运行：
python scripts/evaluate.py --model_dir models/bert-critic-1.0/best_model --csv_path data/6000.csv
"""
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertForSequenceClassification, BertTokenizerFast
from dataset_builder import PromptResponseDataset   # 直接复用之前的 Dataset


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return preds, labels_all


def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存到 {save_path}")


def main(args):
    # -------------- 加载数据（仅使用已标注样本） -------------- #
    df = pd.read_csv(args.csv_path)
    df = df[df["label"].notna()].reset_index(drop=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    ds = PromptResponseDataset(df, tokenizer, args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # -------------- 加载模型 -------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # -------------- 预测 -------------- #
    preds, labels = predict(model, loader, device)

    # -------------- 评估指标 -------------- #
    target_names = ["low (0)", "medium (1)", "high (2)"]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print("\n====== Classification Report ======\n")
    print(report)

    cm = confusion_matrix(labels, preds)

    # -------------- 混淆矩阵可视化 -------------- #
    os.makedirs("results", exist_ok=True)
    plot_confusion(cm, target_names, "results/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="路径：models/bert-base-critic/best_model")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="带标签的 CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
