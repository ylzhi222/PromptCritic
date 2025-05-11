"""
predict.py — 轻量级推理脚本
===================================

支持两种使用方式：
1. 单条输入
python scripts/predict.py --model_dir models/bert-critic-1.0/best_model --prompt  "..." --response "..."

2. 批量 CSV（含 prompt,response 列）
python scripts/predict.py --model_dir models/bert-critic-1.0/best_model --input_csv data/to_score_300.csv --output_csv data/to_score_with_label_300.csv

输出 label: 0=low, 1=medium, 2=high，并附带置信度概率。
"""
import argparse, os, sys, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

def load_model(model_dir: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_single(tokenizer, model, device, prompt: str, response: str):
    text = prompt.strip() + "\n\n" + response.strip()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    label = int(torch.argmax(torch.tensor(probs)).item())
    return label, probs  # probs: [p_low, p_medium, p_high]


def predict_csv(tokenizer, model, device, in_csv: str, out_csv: str):
    df = pd.read_csv(in_csv)
    assert {'prompt', 'response'} <= set(df.columns), "CSV must contain 'prompt' and 'response' columns"
    labels, p0, p1, p2 = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label, probs = predict_single(tokenizer, model, device, row['prompt'], row['response'])
        labels.append(label)
        p0.append(probs[0]); p1.append(probs[1]); p2.append(probs[2])
    df['pred_label'] = labels
    df['prob_low']   = p0
    df['prob_med']   = p1
    df['prob_high']  = p2
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved predictions to {out_csv}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='路径：best_model 文件夹')
    parser.add_argument('--prompt', type=str, help='单条 prompt')
    parser.add_argument('--response', type=str, help='单条 response')
    parser.add_argument('--input_csv', help='批量预测输入 CSV')
    parser.add_argument('--output_csv', help='批量预测输出 CSV')
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_dir)

    # ---- 单条模式 ----
    if args.prompt and args.response:
        label, probs = predict_single(tokenizer, model, device, args.prompt, args.response)
        print(f"Label: {label}  (low=0, medium=1, high=2)")
        print(f"Probabilities: low={probs[0]:.3f}, medium={probs[1]:.3f}, high={probs[2]:.3f}")
        sys.exit(0)

    # ---- 批量模式 ----
    if args.input_csv:
        out_path = args.output_csv or (os.path.splitext(args.input_csv)[0] + '_scored.csv')
        predict_csv(tokenizer, model, device, args.input_csv, out_path)
    else:
        parser.error('必须提供 --prompt+--response 或 --input_csv')

if __name__ == "__main__":
    main()
