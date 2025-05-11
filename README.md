# 🤖 PromptCritic：基于BERT的大语言模型输出质量分类器

PromptCritic 是一个轻量但强大的深度学习工具，用于自动评估大语言模型（LLMs）生成的回答质量。它基于 PyTorch 构建，并从 BERT 微调而来，将回答分类为三个等级 —— **低**、**中** 和 **高**，为提示词优化、数据集筛选和 LLM 评估研究提供了基础。

---

## 📌 功能特性

* ✅ 微调的 BERT 分类器（`bert-base-uncased`）
* ✅ 三类分类：低 (0)、中 (1)、高 (2)
* ✅ 基于 [UltraFeedback 数据集](https://huggingface.co/datasets/openbmb/UltraFeedback)
* ✅ 支持评估、批量预测和可视化
* ✅ 易于重新训练或扩展（支持早停、加权损失）

---

## 🧠 质量评估标准

每个答案根据以下五个因素进行打分：

1. ✅ **是否回答了问题**（是 / 否）
2. 🧠 **事实准确性**（是 / 否，可选用搜索验证）
3. 🗣️ **清晰与简洁度**（好 / 一般 / 差）
4. 🧱 **逻辑结构**（是 / 否）
5. 📏 **篇幅或冗余问题**（是 / 否）

最终标签：

* **0 = 低质量**：臆测、跑题、不连贯
* **1 = 中质量**：部分正确或冗长
* **2 = 高质量**：准确、有帮助、清晰

---

## 🗂 项目结构

```
PromptCritic/
├── data/
│   └── README.md                 # 最终的平衡数据集（每类2000条）
├── models/
│   └── bert-critic-1.0/          # 最终保存的模型（HuggingFace 格式）
├── scripts/
│   ├── convert_ultrafeedback.py  # 将 UltraFeedback 转换为已标注的 CSV
│   ├── dataset_builder.py        # 数据集 + 分词器处理器
│   ├── train.py                  # 基线训练脚本
│   ├── train_V2.py               # 带早停和类别权重的训练
│   ├── evaluate.py               # 模型评估与混淆矩阵
│   ├── predict.py                # 自定义输入或批量推理
│   ├── frequency.py              # 标签频率分析
│   ├── small_sample.py           # 每个标签等量采样
│   └── small_sample_random.py    # 保留标签比例的随机采样
├── results/
│   └── confusion_matrix.png      # 评估可视化图
├── requirements.txt
└── README.md
```

---

## 🧪 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train.py \
    --csv_path data/6000.csv \
    --model_name bert-base-uncased \
    --epochs 4 \
    --batch_size 16 \
    --lr 2e-5 \
    --output_dir models/bert-critic-1.0
```

### 3. 评估模型

```bash
python scripts/evaluate.py \
    --model_dir models/bert-critic-1.0 \
    --csv_path data/6000.csv
```

### 4. 预测

```bash
python scripts/predict.py \
    --model_dir models/bert-critic-1.0 \
    --csv_path your_prompts.csv
```

---

## 🎯 结果展示（基于6000个平衡样本）

| 分类     | 精确率 (Precision) | 召回率 (Recall) | F1 分数         |
| ------ | --------------- | ------------ | ------------- |
| 低 (0)  | 0.764           | 0.863        | 0.811         |
| 中 (1)  | 0.766           | 0.597        | 0.671         |
| 高 (2)  | 0.741           | 0.809        | 0.774         |
| **总体** |                 |              | **0.756 准确率** |

👉 混淆矩阵与分类报告保存在 `results/confusion_matrix.png`。

---

## 🧩 数据集概览

* 来源：[UltraFeedback (openbmb)](https://huggingface.co/datasets/openbmb/UltraFeedback)
* 最终子集：6000 个样本（每个标签2000条）
* 通过脚本 `convert_ultrafeedback.py` 转换并标注
* 平衡且清洗过，适用于可复现训练

---

## 🗃 模型下载

最终模型保存在：

📁 `models/bert-critic-1.0/`
可与 HuggingFace Transformers 一起使用：

```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("models/bert-critic-1.0")
```

---

## 📌 未来计划

* [ ] Streamlit 网页演示（LLM + Critic 互动）
* [ ] 借助 GPT 反馈循环的标注辅助
* [ ] 扩展至多语言回答或 GPT-4 输出
* [ ] 基于回归的细粒度评分

---

## 📜 许可证

本项目遵循 MIT 协议发布。详情请见 `LICENSE` 文件。

---

## 🙏 致谢

* 感谢 [OpenBMB](https://github.com/OpenBMB) 提供 UltraFeedback 数据集
* 感谢 HuggingFace Transformers 提供预训练模型
* 感谢社区成员对 LLM 评估的见解贡献

---

如果你觉得这个项目有用，欢迎 ⭐ 收藏本仓库！
