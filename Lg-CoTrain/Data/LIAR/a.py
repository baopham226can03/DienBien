import pandas as pd

# Đọc TSV và ghi ra CSV
pd.read_csv('val2_final.tsv', sep='\t').to_csv('val.csv', index=False)