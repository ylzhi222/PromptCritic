import pandas as pd

# 读取CSV文件
file_path = r'data\ultrafeedback_promptcritic.csv'
df = pd.read_csv(file_path)

# 统计label列的频数，包含空白格
label_counts = df['label'].isnull().sum()  # 统计空白格的数量
df['label'] = df['label'].fillna('Missing')  # 将空白格填充为 'Missing'，以便统计
label_counts = df['label'].value_counts()  # 统计每个值的出现次数

# 输出结果
print("Label 频数统计：")
print(label_counts)