import pandas as pd

healthData = pd.read_excel('./mydata.xlsx',
                           header = 1,
                           usecols = 'A:Q')

print(healthData.head(3))