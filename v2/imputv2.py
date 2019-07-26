from dictionaries import CCs, NGAs, NON_NUMERIC_COLUMNS
import pandas as pd
import numpy as np
import utm.conversion as utmc
from math import atan2
import datawig
import os
from tqdm import tqdm

# Boolean controlling whether or not to recompute the regression variables for
# each CC. 
RECOMPUTE_REGRESSION_VARS = False

DEBUG = True

def isnan(x):
    if isinstance(x, str):
        return False
    elif isinstance(x, list):
        return x == []
    else:
        return np.isnan(x)

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

def utmCentroid(utm):
    number = utm[:-1]
    letter = utm[-1]
    return (utmc.zone_letter_to_central_latitude(letter),
            utmc.zone_number_to_central_longitude(int(number)))

def getCentroid(utms):
    coords = list(map(utmCentroid,utms))
    [lats, lons] = zip(*coords)
    numCoords = len(coords)
    centerLat = np.sum(lats)/numCoords
    centerLon = np.sum(lons)/numCoords
    return (centerLat, centerLon)
    

def computeDistanceInKm(poli, polj):
    lat1, lon1 = getCentroid(eval(poli['UTM_zone']))
    lat2, lon2 = getCentroid(eval(polj['UTM_zone']))

    # approximate radius of earth in km
    R = 6373.0
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# A sum function that treats nans as 0s
def smartSum(vals):
    return np.sum([v for v in vals if not np.isnan(v)])

def prepareRegressionX1(data, responseVar):
    regression_x1 = []
    # For each polity i
    progBar = tqdm(total=data.shape[0]**2)
    for i, rowi in data.iterrows():
        diffusions = [0]
        # For each other polity j/=i existing at the same time
        for j, rowj in data.iterrows():
            progBar.update(1)
            if j != i and rowi['Period_start'] == rowj['Period_start']:
                # Compute the distance between the polities
                distance = computeDistanceInKm(rowi,rowj)
                # Return exp(-(distance/1100))*responseVar
                diffusions.append(np.exp(-(distance/1100.0))*rowj[responseVar])
        regression_x1.append(smartSum(diffusions))
    data['{}_Regression_x1'.format(responseVar)] = pd.Series(regression_x1) 
    progBar.close()
    return data

def smartMean(vals):
    ls = [l for l in vals if not isnan(l)]
    if len(ls) == 0:
        return np.nan
    return np.mean(ls)

def aggMean(series):
    if series.shape[0] <= 1:
        return series.iloc[0]
    if series.name in NON_NUMERIC_COLUMNS:
        return series.iloc[0]
    else:
        return smartMean(series)

def sumResponse(seshat, ngaInfo, responseVar):
    sum = 0
    prevYear = -20000
    regName = '{}_Regression_x0'.format(responseVar)
    for i, row in ngaInfo.iterrows():
        row[regName] = np.nan
        if row['Period_start'] >= prevYear:
            prevYear = row['Period_start']
            if not isnan(row[responseVar]):
                try:
                    if isnan(seshat.at[row['Temperoculture'], regName]):
                        seshat.at[row['Temperoculture'], regName] = 0
                except:
                    seshat.at[row['Temperoculture'], regName] = 0
                seshat.at[row['Temperoculture'], regName] += sum / len(eval(seshat.at[row['Temperoculture'],'NGA']))
                sum += row[responseVar]
        else:
            seshat.at[row['Temperoculture'], regName] = 0
    return seshat


def prepareRegressionX0(data, responseVar):
    seshat = data.set_index('Temperoculture')
    newRows = []
    # For each NGA
    for nga in NGAs:
        # Fetch info for polities in the NGA
        ngaInfo = data[data['NGA'].map(lambda ngaList: nga in ngaList)]
        # Each polity gets the sum of all temporally previous responseVars
        ngaInfo = ngaInfo.sort_values('Period_start')
        seshat = sumResponse(seshat, ngaInfo, responseVar)
    seshat = seshat.reset_index()
    return seshat


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
    pbar = tqdm(total=data.shape[0]**2)
    for i, rowi in data.iterrows():
        diffusions = [0]
        # For each other polity j/=i existing at the same time
        for j, rowj in data.iterrows():
            pbar.update(1)
            if j != i and rowi['Period_start'] == rowj['Period_start']:
                # Compute the distance between the polities
                diffusions.append(languageFactor(rowi,rowj)*rowj[responseVar])
        regression_x2.append(smartSum(diffusions))
    data['{}_Regression_x2'.format(responseVar)] = pd.Series(regression_x2) 
    pbar.close()
    return data

def prepareModel(seshat, responseVar):
    print('Prior')
    print(seshat.shape)
    print("Preparing {}_Regression_x0...".format(responseVar))
    print(seshat.shape)
    seshat = prepareRegressionX0(seshat, responseVar)
    print("Preparing {}_Regression_x1...".format(responseVar))
    print(seshat.shape)
    seshat = prepareRegressionX1(seshat, responseVar)
    print("Preparing {}_Regression_x2...".format(responseVar))
    print(seshat.shape)
    seshat = prepareRegressionX2(seshat, responseVar)
    print(seshat.shape)
    return seshat

def makeRegressionVars(seshat):
    for i,ccPredict in enumerate(CCs):
        print('Preparing {} ({}/9)...'.format(ccPredict,i+1))
        seshat = prepareModel(seshat, ccPredict)
        seshat.to_csv('model/regression-var-{}.csv'.format(ccPredict))
    seshat.to_csv('model/seshat-with-regression-vars.csv')
    return seshat

# Functional list delete
def lDel(l,x):
    l.remove(x)
    return l

# Scoring function for use with the datawig
def p2Score(true, predicted, confidence):
    return p2prediction(predicted, true)

def modelExists(predictVar):
    return os.path.isdir('model/{}_imputer0'.format(predictVar))


def ccVars(df):
    return [col for col in list(df.columns) if col[:2] == 'CC' and col != 'CCs_imputed']

def getCCTrainSet(seshat):
    trainSet = seshat[ccVars(seshat)]
    return trainSet[trainSet.isnull().sum(axis=1) == 0]

def combineRegWithSeshat(seshat, trainSet):
    seshat.set_index('Temperoculture')
    for regVar in [var for var in list(trainSet.columns) if 'Regression' in var]:
        seshat[regVar] = trainSet[regVar]
    return seshat

def testImpute(data, modelVars):
    train, test = datawig.utils.random_split(data)
    predictVar = 'CC_PolPop'
    actual = test[predictVar].copy()
    test[predictVar] = test[predictVar].map(lambda _ : np.nan)

    imputer = datawig.SimpleImputer(
                input_columns = lDel(modelVars, predictVar),
                output_column = predictVar,
                output_path   = 'model/test_imputer'.format(predictVar)
                )

    imputer.fit_hpo(train_df=train, num_epochs=1000,
            user_defined_scores=[(p2Score, 'p2_prediction')])
    imputed = imputer.predict(test)
    predicted = imputed['{}_imputed'.format(predictVar)]
    print('Pred: {}'.format(p2prediction(predicted,actual)))


def imputeCCs(seshat):
    trainSet    = getCCTrainSet(seshat)
    modelVars   = ccVars(seshat)
    predictData = seshat[modelVars]

    testImpute(trainSet, modelVars)
    exit()

    for predictVar in CCs:
    #    if modelExists(predictVar):
    #        imputer = datawig.SimpleImputer.load('model/{}_imputer'.format(predictVar))
        if True:
            imputer = datawig.SimpleImputer(
                        input_columns = lDel(modelVars, predictVar),
                        output_column = predictVar,
                        output_path   = 'model/{}_imputer'.format(predictVar)
                        )

        imputer.fit_hpo(train_df=trainSet, num_epochs=1000,
                user_defined_scores=[(p2Score, 'p2_prediction')])
        #imputer.load_hpo_model(hpo_name=0)


        seshat[predictVar] = imputer.predict(predictData)['{}_imputed'.format(predictVar)]
    return seshat

def impute(seshat):
    # Keep track of which CC's we're imputing
#    seshat = includeImputeInfo(seshat)
#    seshat = makeRegressionVars(seshat)
    if DEBUG:
        seshat = pd.read_csv('model/seshat-with-regression-vars.csv')
    seshat = imputeCCs(seshat)
    seshat.to_csv('shiny-seshat-imputed.csv')

    return seshat
