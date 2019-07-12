from dictionaries import FEATURES_TO_IMPUTE
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


# Pair down to only datapoints with all 51 variables
def pairDown(fullSeshat):
    seshat = fullSeshat[[
        'CC_PolPop',
        'CC_PolTerr',
        'CC_CapPop',
        'CC_Hier',
        'CC_Govt',
        'CC_Infra',
        'CC_Writing',
        'CC_Texts',
        'CC_Money',
        'NGA',
        'Temperoculture']]
 
    seshat = seshat[seshat.isna().sum(axis=1) == 0]
    return seshat

def impute(fullSeshat):
    # Pair down to only datapoints with all 51 variables
    seshat = pairDown(fullSeshat)
    # Build the full imputation model

    # Use the model to impute the dataset 20 times.

    # (Each imputation introduces variance from sampling the residual to add to the final vals)

    # return recombine()
    seshat['num_NGAs'] = seshat['NGA'].map(len)
    seshat['Temperoculture'] = seshat['Temperoculture'].map(lambda x : x[:-2])
    seshat = seshat.drop_duplicates(subset='Temperoculture')
    print(seshat)
    print(seshat['num_NGAs'].sum())
    return fullSeshat
