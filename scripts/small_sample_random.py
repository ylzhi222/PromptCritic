import pandas as pd

# 读取完整数据
df = pd.read_csv("data/ultrafeedback_promptcritic.csv")
'''
# 抽样比例，例如 5%
frac = 0.05

# 保持比例的随机抽样
df_sampled = df.sample(frac=frac, random_state=42)
'''

#随机抽出300个样本
df_sampled = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=5/len(df), random_state=42)
)


# 保存新文件
df_sampled.to_csv("data/test_5.csv", index=False)
print(df_sampled['label'].value_counts())
