import pandas as pd
import numpy as np
import urllib.request
import os
import os.path
import math
import io
import re
from tqdm import tqdm
from dictionaries import POLITY_ID_REPLACEMENTS, NGA_UTMs, COLUMN_NAME_REMAP,\
        RITUAL_VARIABLES, COLUMN_MERGE, RITUAL_VARIABLE_RENAMES,\
        COLUMN_REORDERING


SESHAT_URL   = 'http://seshatdatabank.info/moralizinggodsdata/data/download.csv'
OUT_FILENAME = 'shiny-seshat.csv'

pd.options.mode.chained_assignment = None  # default='warn'
PROGRESS_BAR = tqdm(total=(3+520))

# Enables a few nice features for development such as intermediate points
DEBUG = True

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
    elif value in [
        'unknown','suspected unknown', 'inferred', 'uncoded', 'present absent'
            ]:
        return np.nan
    else:
        return colValue

def polityIDreplace(id):
    return POLITY_ID_REPLACEMENTS[id]\
            if id in POLITY_ID_REPLACEMENTS.keys() else id

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
                raise Exception('Invalid date: ' + date)

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

# Handler function for managing identically-named variables distinguished 
# primarily by the variable category
def handleDatumAdd(tculture, datum, value):
    # Fetch the variable name
    variableName = datum['Variable']
    # Ritual variables are distinguished primarily by subsection
    if variableName in RITUAL_VARIABLES:
        # Append the subsection name for specificity since we'll be dropping
        # subsection info.
        variableName = datum['Subsection'] + '_' + variableName
    # Add the datum
    tculture[variableName] = value
    return tculture

def addDatum(tcultures, datum, dated=True):
    # Fetch the value from the datum
    value = extractDatumValue(datum)
    # Take advantage of the fact that our data is sorted by Date_From by
    # allowing values to be overwritten if the datum has a newer date
    for tculture in tcultures:
        # Strictly greater than since there will always be a temperoculture
        # existing right at the bounds.
        if not dated or tculture['Period_end'] > datum['Date_From']:
            # Handle identical variable names distinguished by category
            tculture = handleDatumAdd(tculture, datum, value)
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
    PROGRESS_BAR.update(1)
    ngaSlice = seshat[seshat['NGA'] == nga]
    return ngaSlice[ngaSlice['Polity'] == polity]

# Transform Seshat from a long listing of data points into a nice matrix where
# rows are temperocultures (sub-periods of polities) and columns are features
def makeTemperocultureWise(seshat):
    ret = pd.DataFrame(flatten([
        getSubPeriods(getDataSlice(nga, polity, seshat))
        for nga,polity in getNgaPolityPairs(seshat)
    ]))
    if DEBUG:
        ret.to_csv('phase1.csv')
    return ret


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
    

def improveUTMinfo(seshat):
    seshat['UTM_zone'] = seshat['NGA'].map(
            lambda ngaList:
            list(pd.unique(flatten(
                [NGA_UTMs[nga] for nga in ngaList]
                )))
    )
    return seshat

def renameColumns(seshat):
    return seshat.rename(
            lambda x: 
            x.replace(' ','_') if x == 'NGA' or x == 'UTM zone' else
            x[0].capitalize() + x[1:].lower().replace(' ','_').replace('-','')
            ,axis='columns')

def remapColumnNames(name):
    if name in COLUMN_NAME_REMAP.keys():
        return COLUMN_NAME_REMAP[name]
    else:
        return name

def remapRitualVars(name):
    if name in RITUAL_VARIABLE_RENAMES.keys():
        return RITUAL_VARIABLE_RENAMES[name]
    else:
        return name


def convertPeakDate(date):
    if isinstance(date, float):
        try:
            return int(date)
        except:
            return np.nan
    if '-' in date or '–' in date:
        start, end = getDatesFromDuration(date)
        return int((start+end)/2.0)
    else:
        return dateToInt(date)

def isnan(x):
    if isinstance(x, str):
        return False
    elif isinstance(x, list):
        return x == []
    else:
        return np.isnan(x)

# A "smart max" function.
# Returns non-nan values when able.
# If given lists, returns the largest non-nan list.
def notDumbMax(l, r):
    if isnan(l):
        return r
    elif isnan(r):
        return l
    elif isinstance(l, list) and isinstance(r,list):
        if np.isnan(l).any():
            return r
        elif np.isnan(r).any():
            return l
        else:
            return l if len(l) >= len(r) else r
    if isinstance(l,np.ndarray) or isinstance(l,str):
        return l
    else:
        return max(l,r)

# Convenience function for combining two series via maximum
def combineWithMax(df, s1, s2):
    return df[s1].combine(df[s2], notDumbMax)

def combineColumns(seshat):
    for otherCol, baseCol in COLUMN_MERGE.items():
        seshat[baseCol] = combineWithMax(seshat, baseCol, otherCol)
        seshat = seshat.drop(columns=otherCol)
    return seshat

# From StackOverflow user Glen Thompson
# https://stackoverflow.com/questions/24685012
def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

# Delete columns with <0.5% representation in the dataset
def deleteUltraSparse(seshat):
    return seshat.drop(columns=[
	'Religious_tradition',
	'Description_of_the_normative_ideology',
	'Religionfamily',
	'Name_1',
	'Frequency_for_the_participants',
	'Religionsect'
    ])


def phase1Tidy(seshat):
    # Handle tcultures spanning multiple NGAs
    seshat = groupNGAs(seshat)
    # Rename all feature names to be consistent
    seshat = renameColumns(seshat)
    # Include UTM info for all tcultures in a consistent manner
    seshat = improveUTMinfo(seshat)
    # Drop redundant column
    seshat = seshat.drop(columns='Duration')
    # Convert 'Peak date' column to integer dates
    seshat['Peak_date'] = seshat['Peak_date'].map(convertPeakDate)
    # Rename columns with duplicate names
    seshat = df_column_uniquify(seshat)
    # Apply specific column renames
    seshat = seshat.rename(remapColumnNames, axis='columns')
    # Merge disparate columns of the same feature
    seshat = combineColumns(seshat)
    # Another once-over on renaming those darn unwieldy ritual variable names
    seshat = seshat.rename(remapRitualVars, axis='columns')
    # Delete ultra-sparse columns
    seshat = deleteUltraSparse(seshat) 
    # Fix single orphaned value deleted in the above step
    seshat.at[1192,'Most_euphoric_ritual_name'] = 'feast'
    # Finally, reorder the columns into a nice curated order
    seshat = seshat[COLUMN_REORDERING]
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
    if DEBUG:
        seshat = pd.read_csv('phase1.csv',index_col=0)
    seshat = phase1Tidy(seshat)
#    seshat = impute(seshat)
#    seshat = addSC(seshat)
#    seshat = phase2Tidy(seshat)
    export(seshat) 

if __name__ == '__main__':
    main()
