import pandas as pd
df = pd.read_csv('./doc_table.csv')
df.rename(columns={'start_time': 'Opening Time', 'end_time': 'Closing Time'}, inplace=True)
df.to_csv('doc_table.csv', index=False)

