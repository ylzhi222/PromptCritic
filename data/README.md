### 📦 数据获取

本项目使用 [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) 数据集，请自行运行如下命令下载和转换为训练格式：

```bash
#加载UltraFeedback所有数据并处理成csv
python scripts/convert_ultrafeedback.py --split train --out_csv data/ultrafeedback_promptcritic.csv
#从UltraFeedback数据集中取出6000条平衡数据
python scripts/small_sample.py

注：如果你想使用和我一样的数据可以[点击](https://github.com/ylzhi222/PromptCritic/issues/1)
下载。
