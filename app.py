import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- 加载模型 ------------------- #
@st.cache_resource
def load_model(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# ------------------- 单条预测 ------------------- #
def predict_single(prompt, response, tokenizer, model, device):
    text = prompt.strip() + "\n\n" + response.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    label = int(torch.argmax(torch.tensor(probs)).item())
    return label, probs

# ------------------- 页面设置 ------------------- #
st.set_page_config("PromptCritic 可视化", layout="wide")
st.title("🧠 PromptCritic - 回答质量评估器")
model_dir = st.sidebar.text_input("模型目录", value="models/bert-critic-1.0/best_model")
tokenizer, model, device = load_model(model_dir)

# ------------------- 输入区域 ------------------- #
tabs = st.tabs(["📌 单条输入", "📁 批量上传"])

with tabs[0]:
    st.subheader("📌 单条 prompt + response 预测")
    prompt = st.text_area("📝 Prompt", height=120)
    response = st.text_area("💬 Response", height=200)

    if st.button("🔍 预测质量等级"):
        if not prompt or not response:
            st.warning("请填写完整 Prompt 和 Response！")
        else:
            label, probs = predict_single(prompt, response, tokenizer, model, device)
            label_map = {0: "❌ 低质量", 1: "⚠️ 中等质量", 2: "✅ 高质量"}
            st.markdown(f"### 🎯 模型判断结果：{label_map[label]}")
            st.markdown("### 📊 概率分布")
            df_probs = pd.DataFrame({"label": ["低 (0)", "中 (1)", "高 (2)"], "prob": probs})
            fig, ax = plt.subplots()
            sns.barplot(x="label", y="prob", data=df_probs, palette="Blues_d", ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

with tabs[1]:
    st.subheader("📁 批量上传 CSV 文件 (需包含 'prompt', 'response' 列)")
    uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'prompt' not in df.columns or 'response' not in df.columns:
            st.error("❌ CSV 文件中必须包含 'prompt' 和 'response' 列！")
        else:
            st.write("📄 原始数据预览：", df.head())
            labels, probs0, probs1, probs2 = [], [], [], []
            for i, row in stqdm(df.iterrows(), total=len(df), desc="预测中..."):
                label, probs = predict_single(row['prompt'], row['response'], tokenizer, model, device)
                labels.append(label)
                probs0.append(probs[0]); probs1.append(probs[1]); probs2.append(probs[2])
            df['pred_label'] = labels
            df['prob_low'] = probs0
            df['prob_med'] = probs1
            df['prob_high'] = probs2
            st.success("✅ 批量预测完成！")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载结果 CSV", csv, file_name="predicted_results.csv", mime="text/csv")
