from dictionaries import FEATURES_TO_IMPUTE
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp
import os

"""
Algorithm
Pair down the dataset to only cases with fully known 51 variables
For each feature A we're building a model to predict:
    Build a linear regression using all features to predict A
    Select a more refined model using the P-values for each term in the regression
    coeffs, residuals, p^2 = optimize()
        # ^ Here, we get a residual for every value we attempted to predict!

Optimize
Fetch coefficients from the search algorithm
For each train/test set in the 10-fold cross-validation:
    Compute p^2 for the prediction.
Average all P^2 for a master P^2.
P^2 is the fitness function. Will need to flop like silcoeff to turn it into a minimization thing. 


Notes:
    Create each CC
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
    num  = np.sum([(p - a)**2 for p,a in list(zip(predicted,actual))])
    yBar = np.mean(actual)
    dem  = np.sum([(yBar - a)**2 for a in actual])
    return 1.0 - (num/dem)

def regionKFold(seshat):
    return [
        (seshat[seshat['Region'] != reg], seshat[seshat['Region'] == reg])
            for reg in seshat['Region'].unique() 
    ]

def predict(targetVar, parms, trainSet):
    return

def evaluateModel(seshat, targetVar, parms):
    p2s = []
    # For each slice in the k-fold validation
    for trainSet,testSet in regionKFold(seshat):
        # Use the train set to predict the given variable
        predicted = predict(targetVar, parms, trainSet)
        actual    = list(testSet[targetVar])
        # Use the predicted and actual data to get a p2 value
        p2s.append(p2prediction(predicted, actual))
    # Return the average of all p2 values
    return np.mean(p2s)

def makeSearchSpace(targetVar, predictors):
    allVars = ['constant','polPop','polTerr','capPop','hier','govt','infra',
               'writing', 'texts', 'money']
    searchSpace = []
    for var in allVars:
        if var != targetVar and var in predictors:
            searchSpace.append(hp.uniform(var, -5.0, 5.0))
        else:
            searchSpace.append(hp.uniform(var, 0.0, 0.0))
    return tuple(searchSpace)

def optimizeCoeffs(seshat,targetVar,predictors):
    searchSpace = makeSearchSpace(targetVar, predictors)
    return fmin(
        fn=(lambda parms : evaluateModel(seshat, targetVar, parms)),
        space=searchSpace, algo=tpe.suggest, max_evals=1000
        )


def buildModel(seshat):
    predictors = selectPredictors(seshat)
    model = {}
    # For each predictor
    for targetVar,predictors in predictors.iteritems():
        # Find the best coefficients for predicting the variable
        model[targetVar] = optimizeCoeffs(seshat,targetVar,predictors)
    # Return all info found
    return model

# Pair down to only datapoints with all 51 variables
def pairDown(fullSeshat):
    seshat = fullSeshat[CCs]
    seshat = seshat[seshat.isna().sum(axis=1) == 0]
    return seshat

def impute(fullSeshat):
    # Pair down to only datapoints with all CC variables
    seshat = pairDown(fullSeshat)

    # Build the full imputation model
    # Use the model to impute the dataset 20 times.
    model = buildModel(seshat)

    # (Each imputation introduces variance from sampling the residual to add to the final vals)

    # return recombine()
    return seshat
