"""
Fine‑tune BERT/DistilBERT for PromptCritic
=========================================

示例运行：
python scripts/train.py --csv_path data/6000.csv --model_name bert-base-uncased --epochs 6 --batch_size 16 --lr 2e-5 --output_dir models/bert-critic-1.0
"""
import argparse
import os
import torch
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
from dataset_builder import build_loaders


def compute_accuracy(preds, labels):
    preds = preds.argmax(dim=1)
    mask = labels != -1                # 保险起见，忽略 label = -1 的样本
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits.detach(), labels)
    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc = 0, 0
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("label")
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, labels)
    return total_loss / len(loader), total_acc / len(loader)


def main(args):
    # ----------------- Data ----------------- #
    train_loader, val_loader, tokenizer = build_loaders(
        csv_path=args.csv_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # ----------------- Model ---------------- #
    num_labels = 3
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----------------- Optim / Sched -------- #
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ----------------- Train Loop ----------- #
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Train  | loss {train_loss:.4f}  acc {train_acc:.4f}\n"
            f"Val    | loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ Best model saved to {save_path}  (val_acc={val_acc:.4f})")

    print(f"\n🎉 Training finished. Best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="models/bert-critic")
    args = parser.parse_args()
    main(args)
