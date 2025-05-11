import pandas as pd

# 载入完整数据集
df = pd.read_csv("data/ultrafeedback_promptcritic.csv")

# 每类抽样
df_sampled = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=2000, random_state=42))
      .reset_index(drop=True)
)

# 保存小样本 CSV
df_sampled.to_csv("data/6000.csv", index=False)
print("✅ 抽样完成，共", len(df_sampled), "条样本")
