from dictionaries import FEATURES_TO_IMPUTE
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import pandas as pd

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
 

def buildModel(seshat):
#    for cc in CCs:
    predictors = {}
    for cc in CCs:
        target = seshat[cc]
        X = sm.add_constant(seshat.drop(cc,axis='columns'))
        ordinaryLeastSquares  = sm.OLS(target, X)
        results               = ordinaryLeastSquares.fit(transform=False)
        lines = results.summary().as_csv().split('\n')[10:-7]
        with open('temp','w') as f:
            for line in lines:
                f.write(line + '\n')
        res = pd.read_csv('temp',index_col=0)['P>|t| ']
        predictors[cc] = [pred.replace(' ','') for pred, val in res.items() if val < 0.05]
    return seshat

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
    seshat = buildModel(seshat)

    # (Each imputation introduces variance from sampling the residual to add to the final vals)

    # return recombine()
    return seshat
