import pandas as pd
df = pd.read_parquet("gsm8k/main/train-00000-of-00001.parquet")
print(df.head())
