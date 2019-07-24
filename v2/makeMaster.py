import pandas as pd
from dictionaries import CCs

data = pd.read_csv('model/regression_cc1.csv',index_col='Temperoculture')
data = data.rename(columns={
    'Regression_x0':'CC_PolPop_Regression_x0',
    'Regression_x1':'CC_PolPop_Regression_x1',
    'Regression_x2':'CC_PolPop_Regression_x2'
    })

for i in range(8):
    newRegs = pd.read_csv('model/regression_cc{}.csv'.format(i+2),index_col='Temperoculture')
    data['{}_Regression_x0'.format(CCs[i+1])] = newRegs['Regression_x0'].copy()
    data['{}_Regression_x1'.format(CCs[i+1])] = newRegs['Regression_x1'].copy()
    data['{}_Regression_x2'.format(CCs[i+1])] = newRegs['Regression_x2'].copy()

data.to_csv('model/regressions-master.csv')
