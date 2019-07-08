import pandas as pd
import numpy as np
import urllib.request
import os
import os.path
import math
import io
import re
from tqdm import tqdm
#from impute import impute


SESHAT_URL   = 'http://seshatdatabank.info/moralizinggodsdata/data/download.csv'
OUT_FILENAME = 'shiny-seshat.csv'

pd.options.mode.chained_assignment = None  # default='warn'
PROGRESS_BAR = tqdm(total=(3+520))

def downloadSeshat():
    # Set encodings and filenames
    sourceEncoding = 'ISO-8859-3'
    targetEncoding = 'utf8'
    unencodedName  = 'seshat-uncoded.csv'
    encodedName    = 'seshat.csv'

    # Fetch the databank
    print('Downloading Seshat databank...')
    file, headers = urllib.request.urlretrieve(SESHAT_URL, unencodedName)

    # Reencode
    print('Encoding to UTF-8...')
    source      = io.open(file, 'r')
    uncodedData = source.read()
    target      = io.open(encodedName, 'w', encoding=targetEncoding)
    target.write(uncodedData)
    # Remove redundant file
    os.remove(unencodedName)

def removeUndesiredVariables(seshat):
    undesiredVariables = [
            'RA',
            'Alternative names',
            'Expert',
            'Editor',
            'Editer', # Really?,
            'alternate Names of Official Cult',
    ]
    seshat = seshat[~seshat['Variable'].isin(undesiredVariables)]
    return seshat

# Ensure all pre-requisites exist, and download them if they don't.
def ensureReqs():
    if not os.path.isfile('seshat.csv'):
        downloadSeshat()
    return

def getSeshat():
    return pd.read_csv('seshat.csv')

# Function mapped to Value_From to convert from
# string data to integer data
def convertBooleans(colValue):
    value = colValue.lower()
    if value in ['present', 'inferred present', 'inferred inferred present']:
        return 1
    elif value in ['absent','inferred absent']:
        return 0
    elif value in ['unknown','suspected unknown', 'inferred', 'uncoded']:
        return np.nan
    else:
        return colValue

# Dictionary of gross polity name replacements.
# We're obviously not limited to the weird character convention of the other
# polity IDs; we just want some semblance of consistency.
def polityIDreplace(id):
    dict = {
	'Cahokia extra: 1000-2000 CE' : 'CahokiaExtra',
	'Peru Cuzco chiefdom Middle Horizon (650-1000 CE)' : 'CuzcoMidHorizon',
	'Peru Cuzco Valley Killke (1000-1250)' : 'CuzcoValleyKillke1',
	'Peru Cuzco Valley Killke (1250-1400)' : 'CuzcoValleyKillke2',
	'Peru Lucre Basin (1000-1250 CE)' : 'LucreBasin1',
	'Peru Lucre Basin (1300-1400 CE)' : 'LucreBasin2',
	'Iroquois Early Colonial' : 'IrEarlyColonial',
	'Pre-Colonial Finger Lakes' : 'PreColonialFingerLakes',
	'British colonial period and early independent India':'TransitionIndia',
	'Pre-colonial Garo Hills' : 'PreColonialGaroHills',
	'Iceland Commonwealth Period (930-1262 CE)' : 'IcelandCommonwealth',
	'Norway Kingdom' : 'NorwayKingdom',
	'Brooke Raj and Colonial Period' : 'BrookeRaj',
	'Pre-Brooke Raj Period' : 'PreBrookeRaj',
	'Russia Early Russian' : 'RusEarlyRussian',
	'Russia Pre-Russian period' : 'RusPreRussian',
	'Colonial Lowland Andes' : 'ColonialAndes',
	'Eastern Jin' : 'EasternJin',
	'Mali Kingdom of Gao (1080-1236 CE)' : 'KingdomGao',
	'Mali Kingdom of Gao Za Dynasty (700-1080 CE)' : 'KingdomGaoZa',
	'Oro Early Colonial' : 'OroEarlyColonial',
	'Oro Pre-Colonial' : 'OroPreColonial',
	'Early Chinese' : 'EarlyChinese',
	'Late Qing' : 'LateQing',
	'Modern Yemen' : 'ModernYemen',
	'Ottoman Yemen' : 'OttomanYemen' }
    return dict[id] if id in dict.keys() else id

def phase0Tidy(seshat):
    # Replace spaces with underscores to avoid issues with pandas
    seshat.columns = seshat.columns.str.replace(' ','_')
    # Remove undesired variables
    seshat = removeUndesiredVariables(seshat)
    # Remove the asterisks in polity IDs.
    # These asterisks apparently exist so that all Polity IDs are the same
    # number of characters... Except for the ones that aren't. Anyway, they
    # suck and we don't need them.
    seshat['Polity'] = seshat['Polity'].map(lambda x: x.replace('*',''))
    # Rename polities with long gross names 
    seshat['Polity'] = seshat['Polity'].map(polityIDreplace)
    # Remove leading spaces in the value field
    seshat['Value_From'] = seshat['Value_From'].map(
            lambda x: x[1:] if x[0] == ' ' else x
            )
    # Convert booleans from "present/absent" to "1/0"
    seshat['Value_From'] = seshat['Value_From'].map(convertBooleans)
    # Delete stray polity with only two datapoints
    seshat = seshat[seshat['Polity'] != 'UsIroqP']
    return seshat

# Check if a string is a date
def isDate(s):
    return (s[-2:] == 'CE')

# Convert a date string to an integer
def dateToInt(date):
    if isinstance(date, float) or isinstance(date, int):
        return date

    date = date.lower()
    if date[-3:] == 'bce':
        return -1*int(date[:-3])
    elif date[-2:] == 'ce':
        return int(date[:-2]) - 1
    else:
        if isinstance(date, str):
            try:
                return int(date)
            except:
                print(date)
                raise Exception('Invalid date')

# Convert a Duration string to a tuple of year integers
def getDatesFromDuration(eraStr):
    # For polities without a date range, just return nan
    if pd.isnull(eraStr):
        return np.nan,np.nan
    [start, end] = eraStr.replace(' ','').replace('–','-').split('-')
    end = dateToInt(end)
    if isDate(start):
        start = dateToInt(start)
    else:
        start = int(start)*np.sign(end)
        if start > 0:
            start -= 1
    return start,end

# Fetch the list of integer dates from a single column of the polity data
def getDatesFromCol(col, polityData):
    return list(polityData[~polityData[col].isnull()][col].map(dateToInt))

# Fetch a simple list of all dates contained in the polity data
def getAllDates(polityData):
    return getDatesFromCol('Date_From', polityData)\
            + getDatesFromCol('Date_To', polityData)

def getTimePoints(polityData):
    # FIXME TODO Clean this up 
    timePoints = []
    # First, the "Duration" field contains the first and last possible 
    # time points for the polity.
    duration = polityData[polityData['Variable'] == 'Duration'].reset_index()
    empty = duration.empty
    if not empty:
        duration = duration.iloc[0]['Value_From']
        start, end = getDatesFromDuration(duration)
        timePoints = [start,end]
    # Next, simply add all other dates we come across
    timePoints += getAllDates(polityData)
    # Remove duplicate entries and sort
    timePoints = list(pd.unique(timePoints))
    timePoints.sort()
    return timePoints

def initTemperocultures(timePoints, polityData):
    nga, polity = polityData['NGA'].iloc[0], polityData['Polity'].iloc[0]
    tcultures = []
    if len(timePoints) == 0:
        return [pd.Series({'NGA' : nga, 'Temperoculture': polity + '-1'})]
    elif len(timePoints) == 1:
        # FIXME Prettify this code
        return [pd.Series({'NGA' : nga, 'Temperoculture': polity + '-0', 'Period_end':timePoints[0]})]

    for i in range(1,len(timePoints)):
        tcultures.append(pd.Series({
            'NGA' : nga,
            'Temperoculture' : polity + '-' + str(i),
	    'Period_start'   : timePoints[i-1],
	    'Period_end'     : timePoints[i]
        }))
    return tcultures

def extractDatumValue(datum):
    if datum['Value_Note'] == 'range':
        return (float(datum['Value_From']) + float(datum['Value_To']))/2.0
    elif datum['Value_Note'] == 'disputed':
        return np.nan
    else:
        return datum['Value_From']


def addDatum(tcultures, datum, dated=True):
    # Fetch the value from the datum
    value = extractDatumValue(datum)
    # Take advantage of the fact that our data is sorted by Date_From by
    # allowing values to be overwritten if the datum has a newer date
    for tculture in tcultures:
        # Strictly greater than since there will always be a temperoculture
        # existing right at the bounds.
        if not dated or tculture['Period_end'] > datum['Date_From']:
            tculture[datum['Variable']] = value
    return tcultures

def populateTemperocultures(tcultures, polityData):
    # Prepare the data 
    polityData['Date_From'] = polityData['Date_From'].apply(dateToInt)
    polityData['Date_To']   = polityData['Date_To'].apply(dateToInt)
    # Ensure that the data is sorted by Date_From
    polityData = polityData.sort_values('Date_From')
    # For each datum in the polity data
    for i, datum in polityData.iterrows():
        dated = (datum['Date_Note'] in ['simple','range'])
        tcultures = addDatum(tcultures, datum, dated=dated)
    return tcultures

# Given a slice of seshat that contains a single polity and NGA, return all
# temperocultures in the data.
def getSubPeriods(polityData):
    # Identify all time points associated with the polity. The sub-periods are
    # the stretches of time between these dates.
    timePoints = getTimePoints(polityData)
    # Initialize the temperocultures
    tcultures  =  initTemperocultures(timePoints, polityData)
    # Return the populated temperocultures
    return populateTemperocultures(tcultures, polityData)

# Fetch all NGA/Polity pairs 
def getNgaPolityPairs(seshat):
    return pd.unique(list(zip(list(seshat['NGA']), list(seshat['Polity']))))

# Flatten a list of lists into a single list
# E.g., [[1,2,3],[4],[5,6]] -> [1,2,3,4,5,6]
def flatten(l):
    return [item for sublist in l for item in sublist]

# Fetch the part of seshat containing only data for the given NGA/polity pair
def getDataSlice(nga, polity, seshat):
#    PROGRESS_BAR.update(1)
    ngaSlice = seshat[seshat['NGA'] == nga]
    ret = ngaSlice[ngaSlice['Polity'] == polity]
    print(nga,polity)
    print(ret)
    return ret

# Transform Seshat from a long listing of data points into a nice matrix where
# rows are temperocultures (sub-periods of polities) and columns are features
def makeTemperocultureWise(seshat):
    return pd.DataFrame(flatten([
        getSubPeriods(getDataSlice(nga, polity, seshat))
        for nga,polity in getNgaPolityPairs(seshat)
    ]))

# By default, Seshat will store duplicate tcultures with different NGAs listed
# if a tculture happens to span more than one NGA. We change this instead to 
# have unique tcultures simply keep track of a list of NGAs they span.
def groupNGAs(seshat):
    goodNgas = pd.DataFrame(seshat.groupby('Temperoculture').NGA.agg(
        lambda x: list(x)
        ))
    seshat = pd.merge(goodNgas,seshat,on='Temperoculture')
    seshat = seshat.drop('NGA_y',axis=1)
    seshat = seshat.drop_duplicates(subset='Temperoculture')
    seshat = seshat.rename(columns={'NGA_x': 'NGA'})
    return seshat
    

def phase1Tidy(seshat):
    # Handle tcultures spanning multiple NGAs
    seshat = groupNGAs(seshat)
    return seshat

def addSC(seshat):
    return seshat

def phase2Tidy(seshat):
    return seshat

def exportUnimputed(seshat):
    return

def export(seshat):
    seshat.to_csv(OUT_FILENAME, sep=',')

def main():
#    ensureReqs();                            PROGRESS_BAR.update(1)
#    seshat = getSeshat();                    PROGRESS_BAR.update(1)
#    seshat = phase0Tidy(seshat);             PROGRESS_BAR.update(1)
#    seshat = makeTemperocultureWise(seshat)
    seshat = pd.read_csv('phase1.csv',index_col=0)
    seshat = phase1Tidy(seshat)
#    seshat = addSC(seshat)
#    seshat = phase2Tidy(seshat)
#    exportUnimputed(seshat)
#    seshat = impute(seshat)
    export(seshat) 

if __name__ == '__main__':
    main()
