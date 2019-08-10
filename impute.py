from dictionaries import CCs, NGAs, NON_NUMERIC_COLUMNS, IMPUTABLE_VARS
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
    regions = seshat['Region'].map(lambda x: eval(str(x))[0]).unique()
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
    return [1 if null else 0 for null in series]

def imputeDict(series):
    return str(dict(zip(IMPUTABLE_VARS,trackImputes(series))))

def includeImputeInfo(seshat):
    seshat['Percent_CCs_imputed'] = seshat[CCs].isnull().sum(axis=1) / len(CCs)
    seshat['Features_imputed'] = seshat[IMPUTABLE_VARS].isnull().apply(imputeDict, axis=1)
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
    lat1, lon1 = getCentroid(eval(str((poli['UTM_zone']))))
    lat2, lon2 = getCentroid(eval(str((polj['UTM_zone']))))

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
    # Turns out if we're not looking at money or texts, we don't even need x2!
    if responseVar not in ['CC_Money', 'CC_Texts']:
        data['{}_Regression_x1'.format(responseVar)] = pd.Series(np.zeros(data.shape[0]))
        return data

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
                seshat.at[row['Temperoculture'], regName] += sum / len(eval(str(seshat.at[row['Temperoculture'],'NGA'])))
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
    # Turns out if we're not looking at money or texts, we don't even need x2!
    if responseVar not in ['CC_Money', 'CC_Texts']:
        data['{}_Regression_x2'.format(responseVar)] = pd.Series(np.zeros(data.shape[0]))
        return data

    regression_x2 = []
    pbar = tqdm(total=data.shape[0]**2)
    # For each polity i
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
    print("Preparing {}_Regression_x0...".format(responseVar))
    seshat = prepareRegressionX0(seshat, responseVar)
    print("Preparing {}_Regression_x1...".format(responseVar))
    seshat = prepareRegressionX1(seshat, responseVar)
    print("Preparing {}_Regression_x2...".format(responseVar))
    seshat = prepareRegressionX2(seshat, responseVar)
    return seshat

def makeRegressionVars(seshat):
    for i,ccPredict in enumerate(CCs):
        if ccPredict == 'CC_Texts':
            print('Preparing final variables. By necessity, this is an O(n²) process (where n ≈ 1700, the number of temperocultures). Saddle in, this will take a minute.')
        print('Preparing {} ({}/9)...'.format(ccPredict,i+1))
        seshat = prepareModel(seshat, ccPredict)
        seshat.to_csv('model/regression-var-{}.csv'.format(ccPredict))
    return seshat

# Functional list delete
def lDel(l,x):
    m = l.copy()
    m.remove(x)
    return m

# Scoring function for use with the datawig
def p2Score(true, predicted, confidence):
    return p2prediction(predicted, true)

def modelExists(predictVar):
    return os.path.isdir('model/{}_imputer'.format(predictVar))


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
    print(seshat.columns)
    trainSet    = getCCTrainSet(seshat)
    print(trainSet)
    modelVars   = ccVars(seshat)
    for predictVar in CCs:
        predictData = seshat[modelVars]
        predictData = predictData[predictData[predictVar].isnull()]
        if modelExists(predictVar):
            imputer = datawig.SimpleImputer.load('model/{}_imputer'.format(predictVar))
            imputer.load_hpo_model(hpo_name=0)
        else:
            imputer = datawig.SimpleImputer(
                        input_columns = lDel(modelVars, predictVar),
                        output_column = predictVar,
                        output_path   = 'model/{}_imputer'.format(predictVar)
                        )
            imputer.fit_hpo(train_df=trainSet, num_epochs=1000,
                    user_defined_scores=[(p2Score, 'p2_prediction')])


        pred = imputer.predict(predictData)['{}_imputed'.format(predictVar)]
        seshat[predictVar] = pd.concat([seshat[predictVar].dropna(), pred,]).reindex_like(seshat[predictVar])
    return seshat

def firstImpute(seshat):
    # Keep track of which variables we're imputing
    seshat = includeImputeInfo(seshat)
    if not DEBUG:
        seshat = makeRegressionVars(seshat)
        seshat.to_csv('model/seshat-with-regression-vars.csv')
        seshat = imputeCCs(seshat)
        seshat.set_index('Temperoculture').to_csv('shiny-seshat-CCs-imputed.csv')
    else:
        #seshat = pd.read_csv('model/seshat-with-regression-vars.csv')
        seshat = pd.read_csv('shiny-seshat-CCs-imputed.csv')
    return seshat

def secondImpute(seshat):
    # Already imputed the CCs, so just grab everything else that is imputable
    varsToImpute = [v for v in IMPUTABLE_VARS if v not in CCs]
    for predictVar in tqdm(varsToImpute):
        print("Imputing: {}".format(predictVar))
        imputeData = seshat[IMPUTABLE_VARS]
        # Train set is all of the entries where the target column is not null
        trainSet = imputeData[~imputeData[predictVar].isnull()]
        # And the prediction set is everything else
        predictSet = imputeData[imputeData[predictVar].isnull()]
        # If the training set is the entire set, we've hit a CC-related var we've
        # already imputed, so just skip this feature
        if trainSet.shape[0] == seshat.shape[0]:
            continue
        modelPath = 'model/{}_imputer'.format(predictVar.replace('/',''))
        if modelExists(predictVar):
            imputer = datawig.SimpleImputer.load(modelPath)
            imputer.load_hpo_model(hpo_name=0)
        else:
            imputer = datawig.SimpleImputer(
                        input_columns = lDel(IMPUTABLE_VARS, predictVar),
                        output_column = predictVar,
                        output_path   = modelPath
                        )
            imputer.fit(train_df=trainSet, num_epochs=1000)

        predicted = imputer.predict(predictSet)
        pred = predicted['{}_imputed'.format(predictVar)]
        seshat[predictVar] = pd.concat([seshat[predictVar].dropna(), pred,]).reindex_like(seshat[predictVar])
    return seshat
