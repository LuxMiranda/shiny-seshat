from dictionaries import IMPUTABLE_VARS,IMPUTABLE_CATEGORICAL_VARS, CCs
import pandas as pd
import datawig
from sklearn.metrics import r2_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from impute import p2prediction
import numpy as np
from impute import regionKFold
import os

def p2true(predicted, actual):
    yBar = np.mean(actual)
    dem  = np.sum([(yBar - a)**2 for a in actual])
    if dem == 0:
        return num/yBar
    else:
        num  = np.sum([(p - a)**2 for p,a in list(zip(predicted,actual))])
        return 1.0 - (num/dem)


# Functional list delete
def lDel(l,x):
    m = l.copy()
    m.remove(x)
    return m

def score(true, pred):
    return r2_score(true,pred), p2prediction(pred,true), p2true(pred,true)

def ccVars(df):
    return [col for col in list(df.columns) if col[:2] == 'CC' and col != 'CCs_imputed']

def crossValKFold(df, k):
    shuffled = df.sample(frac=1)
    foldSize = df.shape[0]/k
    folds = []
    for i in range(k):
        begin = int(i*foldSize)
        end   = int((i+1)*foldSize)
        testset  = shuffled.iloc[begin:end]
        trainset = pd.concat([shuffled.iloc[:begin], shuffled.iloc[end:]])
        folds.append((i,trainset,testset))
    return folds

def main():
    seshat = pd.read_csv('model/seshat-with-regression-vars.csv')
    seshat = seshat.groupby(['BasePolity']).first()
    betterWithAllVars = ['CC_Govt','CC_Hier','CC_Infra','CC_Money','CC_Texts','CC_Writing']
    modelVars   = ccVars(seshat)
    # For each imputable variable
    #for predictVar in IMPUTABLE_VARS:
    for predictVar in CCs:
        r2s = []
        p2s = []
        p2ms = []
        varSet = 'few'
#        if predictVar in betterWithAllVars:
#            varSet = 'many'
#            modelVars = IMPUTABLE_VARS
        print('Validating {}'.format(predictVar))
        # Select known values
        knownVals = (seshat[~seshat[predictVar].isna()])
        # Generate a train and test set on known values

#        for df_train, df_test, i in regionKFold(knownVals):
        for i, df_train, df_test in crossValKFold(knownVals, 5):
            # Train a model using the train set
            modelPath = 'model/test_{}_{}_imputer_{}'.format(i,predictVar.replace('/',''),varSet)
            if os.path.isdir(modelPath):
                imputer = datawig.SimpleImputer.load(modelPath)
                imputer.load_hpo_model(hpo_name=0)
            else: 
                imputer = datawig.SimpleImputer(
                        input_columns = lDel(modelVars, predictVar),
                        output_column = predictVar,
                        output_path   = modelPath
                        )
                imputer.fit_hpo(train_df=df_train, num_epochs=1000)
            # Predict the values in the test set
            predicted = imputer.predict(df_test)
            if predictVar in IMPUTABLE_CATEGORICAL_VARS:
                p,r,f,s = precision_recall_fscore_support(predicted[predictVar],predicted['{}_imputed'.format(predictVar)])
                with open('validationCategorical.csv', 'a') as f:
                    f.write('{},{},{}\n'.format(predictVar,r2,p2))

            else:
                try:
                # Compute fidelity metrics
                    r2,p2m,p2 = score(
                        np.array(predicted[predictVar]).astype(np.float64),
                        np.array(predicted['{}_imputed'.format(predictVar)]).astype(np.float64)
                        )
                    r2s.append(r2)
                    p2s.append(p2)
                    p2ms.append(p2m)
                    with open('validationRegression.csv', 'a') as f:
                        f.write('{},{},{},{}\n'.format(i,predictVar,r2,p2m,p2))
                except:
                    continue
        with open('final.csv','a') as f:
            f.write('{},{},{},{}\n'.format(predictVar,np.mean(r2s),np.mean(p2s),np.mean(p2ms)))


def test():
    a = pd.DataFrame(list('abcdefghijklmnopqrstuvwxyz'))
    for i,tr,te in(crossVal10Fold(a)):
        print('Train:\n {}'.format(tr))
        print('Test:\n {}'.format(te))

if __name__ == '__main__':
    main()
