### 📦 数据获取

本项目使用 [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) 数据集，请自行运行如下命令下载和转换为训练格式：

```bash
python scripts/convert_ultrafeedback.py --split train --out_csv data/6000.csv --take_all False
