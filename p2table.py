import pandas as pd

data = pd.read_csv('ccvals.csv')

fews  = data[data['regVars'] == 'few']
manys = data[data['regVars'] == 'many']

grp = lambda df: df.groupby(['CC']).mean()
fews = grp(fews)
manys = grp(manys)

print(fews)
print(manys)
