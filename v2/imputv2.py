from dictionaries import FEATURE_SELECT, NGA_UTMs, CCs
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
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
import utm.conversion as utmc
import math

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

ESTIMATORS = {
#    'LinearRegression'     : lm.LinearRegression(),
#    'Lasso'                : lm.Lasso(),
#    'LassoCV'              : lm.LassoCV(),
#    'ExtraTreesRegressor'  : ens.ExtraTreesRegressor(n_estimators=10, random_state=0),
    'BayesianRidge'        : lm.BayesianRidge(),
#    'Huber'        : lm.HuberRegressor(max_iter=1000),
#    'Huber2'        : lm.HuberRegressor(max_iter=1000,epsilon=1.01),
    'Lasso' : lm.Lasso(),
    'ElasticNet' : lm.ElasticNet(),
#    'LassoLarsIC'          : lm.LassoLarsIC(criterion='aic', fit_intercept=True),
#    'Poly2' : Pipeline([('poly', PolynomialFeatures(degree=2)),
#                        ('linear', lm.BayesianRidge())]),
#    'PolyHuber' : Pipeline([('poly', PolynomialFeatures(degree=2)),
#                        ('linear', lm.HuberRegressor())]),
#
#
#    'PolyRidge' : Pipeline([('poly', PolynomialFeatures(degree=2)),
#                        ('linear', lm.BayesianRidge())]),
#    'PolyTrees' : Pipeline([('poly', PolynomialFeatures(degree=2)),
#                        ('linear', ens.ExtraTreesRegressor())]),
#    'PAR' : lm.PassiveAggressiveRegressor(),
#
#    'Poly3' : Pipeline([('poly', PolynomialFeatures(degree=3)),
#                        ('linear', lm.LinearRegression(fit_intercept=False))]),
#    'Poly4' : Pipeline([('poly', PolynomialFeatures(degree=4)),
#                        ('linear', lm.LinearRegression(fit_intercept=False))]),
#    'Poly5' : Pipeline([('poly', PolynomialFeatures(degree=5)),
#                        ('linear', lm.LinearRegression(fit_intercept=False))]),
#
#
#
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
    regions = seshat['Region'].map(lambda x: eval(x)[0]).unique()
    regionSelector = lambda reg : seshat['Region'].map(lambda l: reg in l)
    return [
        (
            seshat[~regionSelector(reg)],
            seshat[regionSelector(reg)],
            reg
        )
        for reg in regions
    ]

# Pair down to only datapoints with all CCs
def pairDown(fullSeshat):
    ccs = fullSeshat[CCs]
    pairedDown = fullSeshat[ccs.isna().sum(axis=1) == 0]
    pairedDown.to_csv('training/trainingData.csv')
    return pairedDown

def trackImputes(series):
    return [1 if null else 0 for null in series.isnull()]

def includeImputeInfo(seshat):
    seshat['Percent_CCs_imputed'] = seshat[CCs].isnull().sum(axis=1) / len(CCs)
    seshat['CCs_imputed'] = seshat[CCs].apply(trackImputes, axis=1)
    return seshat

def featureSelect(train,test,ccPredict):
    return train[FEATURE_SELECT[ccPredict]], test[FEATURE_SELECT[ccPredict]]

def sumResponse(ngaInfo, responseVar):
    sum = 0
    prevYear = -20000
    newRows = []
    for i, row in ngaInfo.iterrows():
        if row['Period_start'] > prevYear:
            prevYear = row['Period_start']
            row['Regression_x0'] = sum
            sum += row[responseVar]
        else:
            row['Regression_x0'] = 0
        newRows.append(row)
    return newRows

def computeDistanceInKm(poli, polj):
    # approximate radius of earth in km
    R = 6373.0

    #TODO maybe calculate the centroid of all UTM zones a polity spans?
    zonei = eval(poli['UTM_zone'])[0]
    zonej = eval(polj['UTM_zone'])[0]

    num1, let1 = zonei[:-1], zonei[-1]
    num2, let2 = zonej[:-1], zonej[-1]

    lat1 = utmc.zone_letter_to_central_latitude(let1)
    lon1 = utmc.zone_number_to_central_longitude(int(num1))
    lat2 = utmc.zone_letter_to_central_latitude(let2)
    lon2 = utmc.zone_number_to_central_longitude(int(num2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def prepareRegressionX1(data, responseVar):
    # For each polity i
    regression_x1 = []
    for i, rowi in data.iterrows():
        diffusions = [0]
        # For each other polity j/=i existing at the same time
        for j, rowj in data.iterrows():
            if j != i and rowi['Period_start'] == rowj['Period_start']:
                # Compute the distance between the polities
                distance = computeDistanceInKm(rowi,rowj)
                # Return exp(-(distance/1100))*responseVar
                diffusions.append(np.exp(-(distance/1100.0))*rowi[responseVar])
        regression_x1.append(np.sum(diffusions))
    data['Regression_x1'] = pd.Series(regression_x1) 
    return data

def prepareRegressionX0(data, responseVar):
    ngas = pd.Series(np.concatenate(list(data['NGA']))).unique()
    newRows = []
    # For each NGA
    for nga in ngas:
        # Fetch info for polities in the NGA
        ngaInfo = data[data['NGA'].map(lambda ngaList: nga in ngaList)]
        # Each polity gets the sum of all temporally previous responseVars
        ngaInfo = ngaInfo.sort_values('Period_start')
        newRows.append(sumResponse(ngaInfo, responseVar))

    newInfos = pd.DataFrame(np.concatenate(newRows),columns=list(data.columns)+['Regression_x0'])
    return newInfos

def languageFactor(poli, polj):
    if poli['Language'] == polj['Language']:
        return 1
    elif poli['Linguistic_family'] == polj['Linguistic_family']:
        return 0.25
    else:
        return 0

def prepareRegressionX2(data, responseVar):
    # For each polity i
    regression_x2 = []
    for i, rowi in data.iterrows():
        diffusions = [0]
        # For each other polity j/=i existing at the same time
        for j, rowj in data.iterrows():
            if j != i and rowi['Period_start'] == rowj['Period_start']:
                # Compute the distance between the polities
                diffusions.append(languageFactor(rowi,rowj)*rowi[responseVar])
        regression_x2.append(np.sum(diffusions))
    data['Regression_x2'] = pd.Series(regression_x2) 
    return data


def prepareModel(data, responseVar):
    data['NGA'] = data['NGA'].map(eval)
    print("Preparing regression variable x0...")
    model = prepareRegressionX0(data, responseVar)
    print("Preparing regression variable x1...")
    model = prepareRegressionX1(model, responseVar)
    print("Preparing regression variable x2...")
    model = prepareRegressionX2(model, responseVar)
    print(model['Regression_x0'])
    return model

def selectPredictors(data,responseVar):
    data = data[FEATURE_SELECT[responseVar] + ['Regression_x0','Regression_x1','Regression_x2']]
    return data

def testMultImputation(train, test, ccPredict, ccActual, estimator):
    # Perform imputation 20 times and take the average p2 of all of them.
    num_imputations = 1
    p2s = []

    train = selectPredictors(train, ccPredict)
    test  = selectPredictors(test, ccPredict)


    for i in range(num_imputations):
        imputer = IterativeImputer(
                    estimator=ESTIMATORS[estimator],
                    max_iter=10,
                    min_value=0, 
                    random_state=i,
                    sample_posterior=False,
                    verbose=0)

        # Janky workaround for sklearn bug affecting .fit().transform()
        allData     = np.concatenate([np.array(train),np.array(test)])
        imputedData = imputer.fit_transform(allData)
 
        predicted = imputedData[train.shape[0]:,FEATURE_SELECT[ccPredict].index(ccPredict)]
        p2s.append(p2prediction(predicted, ccActual.copy()))

    return np.mean(p2s)

def testImputationModel(ccDataOrig, estimator):
    meanp2s = []

    for i,ccPredict in enumerate(CCs):
        print('Testing {} (CC {}/9)...'.format(estimator,i+1))

#        ccData = ccDataOrig.copy()
#        ccData = prepareModel(ccData, ccPredict)
#        ccData.to_csv('model/regression_cc{}.csv'.format(i+1))

        ccData = pd.read_csv('model/regression_cc{}.csv'.format(i+1), index_col='Temperoculture')
    
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
