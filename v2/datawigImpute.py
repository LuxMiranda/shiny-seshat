import datawig
import pandas as pd
import numpy as np
from imputv2 import p2prediction, regionKFold
from dictionaries import CCs
from pymonad.Reader import curry

def listWithout(l, x):
    print(l)
    temp = l.copy()
    temp.remove(x)
    return temp


def myScore(true, predicted, confidence):
    return p2prediction(predicted,true)

def main():
    df = pd.read_csv('model/seshat-with-regression-vars.csv',index_col='Temperoculture')
    modelVars = [col for col in list(df.columns) if col[:2] == 'CC']
    #modelVars = CCs
    df = df[modelVars]
    df = df[df.isnull().sum(axis=1) == 0]
    df_train_all, df_test_all = datawig.utils.random_split(df)
    #folds = regionKFold(df)
    folds = []
    folds += [(df_train_all,df_test_all,'all regions')]

    #modelVars = CCs


    for var in CCs:
        if True:
            with open('crossValResults.txt','a') as f:
                p2s = []
                f.write('---{}---\n'.format(var))
                for df_train,df_test,region in folds:
                    if region == 'all regions':

                        predictVar = var

                        actual = df_test[predictVar].copy()

                        df_test[predictVar] = df_test[predictVar].map(lambda _ : np.nan)

                        imputer = datawig.SimpleImputer(
                                input_columns=listWithout(modelVars,predictVar),
                                output_column=predictVar,
                                output_path='model/imputer_model_datawig'
                                )
                        imputer.fit_hpo(train_df=df_train, num_epochs=1000,
                                user_defined_scores=[(myScore, 'p2_prediction')])
                        imputed = imputer.predict(df_test)
                        predicted = imputed['{}_imputed'.format(var)]

                        p2s.append((region,p2prediction(predicted,actual)))

                for region,p2 in p2s:
                    f.write('p2 for {}: {}\n'.format(region,p2))

    return

if __name__ == '__main__':
    main()
