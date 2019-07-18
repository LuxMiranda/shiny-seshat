from dictionaries import FEATURE_SELECT
import sklearn.linear_model as lm
import sklearn.ensemble as ens
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sklearn.feature_selection as fs
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp
import os
import sys

"""
Predictive power test

Pair down data to only cases with complete CCs

For each model M we wish to test:
    For each variable Q we wish to predict:
        Split the data into train and test set
        Remove the Q data from the test set
        Fit an imputer on the train set using model M
        Multiple-impute the test set
        Get the average p2 from the multiple imputation
    model M has p2 values for these variables
    
"""

CCs = [ 'CC_PolPop',
        'CC_PolTerr',
        'CC_CapPop',
        'CC_Hier',
        'CC_Govt',
        'CC_Infra',
        'CC_Writing',
        'CC_Texts',
        'CC_Money']

ESTIMATORS = {
    'LinearRegression'     : lm.LinearRegression(),
#    'Lasso'                : lm.Lasso(),
#    'LassoCV'              : lm.LassoCV(),
#    'ExtraTreesRegressor'  : ens.ExtraTreesRegressor(n_estimators=10, random_state=0),
#    'BayesianRidge'        : lm.BayesianRidge(),
#    'LassoLarsIC'          : lm.LassoLarsIC(criterion='aic', fit_intercept=True),
#    'RANSAC'               : lm.RANSACRegressor(),
#    'RidgeCV'              : lm.RidgeCV(),                                                               
}

# Really, REALLY janky way of going about parsing p-values from the
# regression result. But hey, it works. Returns a list of significant coeffs
def parseResults(fit):
    # Split the summary string into lines
    lines = fit.summary().as_csv().split('\n')[10:-7]
    # Write to a temporary file to create a proper csv
    with open('temp','w') as f:
        for line in lines:
            f.write(line + '\n')
    # Now, read that csv into a dataframe and select the P values
    res = pd.read_csv('temp',index_col=0)
    pvals = res['P>|t| ']
    # Clean up the temp ile
    os.remove('temp')
    # Turn it into a pretty list of significant coefficients with initial guesses
    return [pred.replace(' ','') for pred, val in pvals.items() if val < 0.05]


# Use a least-squares regression to select an appropriate model by using
# p-values to determine the most significant terms to include 
def selectPredictors(seshat):
    predictors = {}
    for cc in CCs:
        target = seshat[cc]
        # Add all terms but the target term
        X = sm.add_constant(seshat.drop(cc,axis='columns'))
        # Compute a regression model using least squares
        ordinaryLeastSquares  = sm.OLS(target, X)
        fit                   = ordinaryLeastSquares.fit(transform=False)
        predictors[cc]        = parseResults(fit)
    return predictors

def p2prediction(predicted, actual):
    yBar = np.mean(actual)
    dem  = np.sum([(yBar - a)**2 for a in actual])
    if dem == 0:
        num = np.sum([-np.abs(p - yBar) for p in predicted])
        return num/yBar + 1
    else:
        num  = np.sum([(p - a)**2 for p,a in list(zip(predicted,actual))])
        return 1.0 - (num/dem)

def regionKFold(seshat):
    regions = seshat['Region'].apply(lambda x: x[0]).unique()
    regionSelector = lambda reg : seshat['Region'].map(lambda l: reg in l)
    return [
        (
            seshat[~regionSelector(reg)],
            seshat[regionSelector(reg)],
            reg
        )
        for reg in regions
    ]

# Pair down to only datapoints with all 51 variables
def pairDown(fullSeshat):
    ccs = fullSeshat[CCs]
    return fullSeshat[ccs.isna().sum(axis=1) == 0]

def trackImputes(series):
    return [1 if null else 0 for null in series.isnull()]

def includeImputeInfo(seshat):
    seshat['Percent_CCs_imputed'] = seshat[CCs].isnull().sum(axis=1) / len(CCs)
    seshat['CCs_imputed'] = seshat[CCs].apply(trackImputes, axis=1)
    return seshat

def featureSelect(train,test,ccPredict):
    return train[FEATURE_SELECT[ccPredict]], test[FEATURE_SELECT[ccPredict]]

def prepareModel(data, responseVar):
    print('')
    print(data)
    exit()

def testMultImputation(train, test, ccPredict, ccActual, estimator):
    # Perform imputation 20 times and take the average p2 of all of them.
    num_imputations = 1
    p2s = []

    for i in range(num_imputations):
        imputer = IterativeImputer(
                    estimator=ESTIMATORS[estimator],
                    max_iter=10,
                    min_value=0, 
                    random_state=i,
                    sample_posterior=False,
                    verbose=0)

        # Prepare models
        train = prepareModel(train,ccPredict)
        test  = prepareModel(test,ccPredict)

        # Janky workaround for sklearn bug affecting .fit().transform()
        allData     = np.concatenate([np.array(train),np.array(test)])
        imputedData = imputer.fit_transform(allData)
 
        predicted = imputedData[train.shape[0]:,FEATURE_SELECT[ccPredict].index(ccPredict)]
        p2s.append(p2prediction(predicted, ccActual.copy()))

    return np.mean(p2s)

def testImputationModel(ccData, estimator):
    meanp2s = []

    for i,ccPredict in enumerate(CCs):
        sys.stdout.write('\rTesting {} (CC {}/9)...'.format(estimator,i+1))
        sys.stdout.flush()

        p2s = []
        for train, test, testRegion in regionKFold(ccData):
            # Remove the cc we're predicting in the test set
            ccActual = test[ccPredict].copy()
            test[ccPredict] = test[ccPredict].map(lambda x: np.nan)
            p2s.append(testMultImputation(train, test, ccPredict, ccActual, estimator))
        meanp2s.append(np.mean(p2s))

    print('')

    return meanp2s

def testImputationModels(seshat):
    ccData  = pairDown(seshat)
    p2data  = []
    for estimator in ESTIMATORS.keys():
        p2data.append(
            testImputationModel(ccData, estimator) + [estimator]
        )
        pd.DataFrame(p2data, columns=(CCs+['Estimator'])).to_csv('estimator-predictions.csv',mode='a',header=False)
    return pd.DataFrame(p2data, columns=(CCs + ['Estimator']))

def impute(seshat):
    # Keep track of which CC's we're imputing
    seshat = includeImputeInfo(seshat)

    # Test imputation methods
    results = testImputationModels(seshat)
    results = results.set_index('Estimator')
    print(results)


    return seshat
