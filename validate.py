from dictionaries import IMPUTABLE_VARS,IMPUTABLE_CATEGORICAL_VARS, CCs
import pandas as pd
import datawig
from sklearn.metrics import r2_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from impute import p2prediction
import numpy as np
from impute import regionKFold
import os

# Functional list delete
def lDel(l,x):
    m = l.copy()
    m.remove(x)
    return m

def score(true, pred):
    return r2_score(true,pred), p2prediction(pred,true)

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
    seshat = seshat.groupby(['BasePolity']).mean()
    #modelVars = IMPUTABLE_VARS
    modelVars = ccVars(seshat)

    # For each imputable variable
    #for predictVar in IMPUTABLE_VARS:
    for predictVar in CCs:
        print('Validating {}'.format(predictVar))
        # Select known values
        knownVals = (seshat[~seshat[predictVar].isna()])
        # Generate a train and test set on known values

#        for df_train, df_test, i in regionKFold(knownVals):
        for i, df_train, df_test in crossValKFold(knownVals, 5):
            # Train a model using the train set
            modelPath = 'model/test_{}_{}_imputer_a'.format(i,predictVar.replace('/',''))
            if os.path.isdir(modelPath):
                imputer = datawig.SimpleImputer.load(modelPath)
                imputer.load_hpo_model(hpo_name=0)
            else: 
                imputer = datawig.SimpleImputer(
                        input_columns = lDel(modelVars, predictVar),
                        output_column = predictVar,
                        output_path   = modelPath
                        )
                imputer.fit(train_df=df_train, num_epochs=1000)
            # Predict the values in the test set
            predicted = imputer.predict(df_test)
            if predictVar in IMPUTABLE_CATEGORICAL_VARS:
                p,r,f,s = precision_recall_fscore_support(predicted[predictVar],predicted['{}_imputed'.format(predictVar)])
                with open('validationCategorical.csv', 'a') as f:
                    f.write('{},{},{}\n'.format(predictVar,r2,p2))

            else:
                try:
                # Compute fidelity metrics
                    r2,p2 = score(
                        np.array(predicted[predictVar]).astype(np.float64),
                        np.array(predicted['{}_imputed'.format(predictVar)]).astype(np.float64)
                        )
                    with open('validationRegression.csv', 'a') as f:
                        f.write('{},{},{},{}\n'.format(i,predictVar,r2,p2))
                except:
                    continue

def test():
    a = pd.DataFrame(list('abcdefghijklmnopqrstuvwxyz'))
    for i,tr,te in(crossVal10Fold(a)):
        print('Train:\n {}'.format(tr))
        print('Test:\n {}'.format(te))

if __name__ == '__main__':
    main()
