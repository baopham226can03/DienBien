import pandas as pd
df = pd.read_csv('du_pseudo_phi3.csv')
print(df['pseudo_label'].unique())