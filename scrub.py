import pandas as pd
import numpy as np
import urllib.request
import os
import os.path
import math
import io
import re
from math import log10
from impute import firstImpute, secondImpute
from tqdm import tqdm
from dictionaries import POLITY_ID_REPLACEMENTS, NGA_UTMs, COLUMN_NAME_REMAP,\
        RITUAL_VARIABLES, COLUMN_MERGE, RITUAL_VARIABLE_RENAMES,\
        COLUMN_REORDERING, NGA_REGIONS

SESHAT_URL   = 'http://seshatdatabank.info/moralizinggodsdata/data/download.csv'
OUT_FILENAME = 'shiny-seshat.csv'
OUT_UNRESOLVED_FILENAME = 'shiny-seshat-unresolved.csv'

pd.options.mode.chained_assignment = None  # default='warn'
print('Performing initial clean-up...')
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
            'Editer', # Nice
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

# Function mapped to Value_From and Value_To to convert from
# string data to integer data
def convertBooleans(colValue):
    value = colValue
    if isinstance(value, str):
        value = value.lower()
    if value in ['present', 'inferred present', 'inferred inferred present', 'present]']:
        return 1.0
    elif value in ['inferred present','inferred inferred present']:
        return 0.9
    elif value in ['absent','none']:
        return 0.0
    elif value in ['inferred absent']:
        return 0.1
    elif value in ['present absent','inferred']:
        return 0.5
    elif value in [
        'unknown','suspected unknown', 'uncoded', 'suspected unknwon',
        'suspect unknown', 'suspected unkown']:
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
    seshat['Value_To'] = seshat['Value_From'].map(convertBooleans)
    # Delete stray polity with only two datapoints
    seshat = seshat[seshat['Polity'] != 'UsIroqP']
    return seshat

# check if a string is a date
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
    if datum['Value_Note'] in ['range','disputed']\
            and isinstance(datum['Value_From'],float):
        return (float(datum['Value_From']) + float(datum['Value_To']))/2.0
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

NUM_POPS = 0

def populateTemperocultures(tcultures, polityData):
    global NUM_POPS
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


# Fix singular orphaned values arisen from human error in encoding Seshat.
def findHomesForThePoorOrphanChildren(seshat):
    # This little child was so busy feasting, they drifted into a deleted column
    seshat.at[1192,'Most_euphoric_ritual_name'] = 'feast'
    # This little child was ~very~ found of extraneous units
    seshat.at[441, 'Polity_territory'] = 175000
    # This pair of little twins never really got a hang of intervals in math class
    seshat.at[49, 'Population_of_the_largest_settlement'] = 22500
    seshat.at[50, 'Population_of_the_largest_settlement'] = 22500
    # This trio of triplets decided to skip 19th-century day in history class
    seshat.at[954, 'Period_start'] = 1825
    seshat.at[954, 'Period_end']   = 1830
    seshat.at[955, 'Period_start'] = 1830
    # Seven silly stays have some six or seven serious administrative levels
    seshat.at[954, 'Administrative_levels'] = 6.5
    seshat.at[955, 'Administrative_levels'] = 6.5
    seshat.at[956, 'Administrative_levels'] = 6.5
    seshat.at[957, 'Administrative_levels'] = 6.5
    seshat.at[958, 'Administrative_levels'] = 6.5
    seshat.at[959, 'Administrative_levels'] = 6.5
    seshat.at[960, 'Administrative_levels'] = 6.5
    # This orphan never ever wanted to be adopted, so much so that they posted
    # a big sign outside the orphanage indicating so. Unfortunately, this 
    # advertising led to the orphan receiving an extra amount of attention
    # from potential adopters. Let's help the orphan out by removing the sign.
    seshat = seshat.set_index('Temperoculture')
    seshat.at['PkProto-1', 'Largest_communication_distance'] = np.nan
    # More extraneous units
    seshat.at['CnWHan-1', 'Monumental_building_extent'] = 2400
    seshat.at['CnWHan-2', 'Monumental_building_extent'] = 2400
    seshat.at['CnWHan-3', 'Monumental_building_extent'] = 2400
    seshat.at['CnWHan-4', 'Monumental_building_extent'] = 2400
    return seshat.reset_index()

def addRegionInfo(seshat):
    seshat['Region'] = seshat['NGA'].map(
            lambda ngaList:
            list(pd.unique([NGA_REGIONS[nga] for nga in ngaList]))
    )
    return seshat

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
    # Fix single orphaned values that arose from human error in coding Seshat
    seshat = findHomesForThePoorOrphanChildren(seshat)
    # Add Region
    seshat = addRegionInfo(seshat)
    # Finally, reorder the columns into a nice curated order
    seshat = seshat[COLUMN_REORDERING]
    return seshat


def log10float(x):
    return log10(float(x))

# CC_PolPop
#   Log_10 of the population.
def createCC_PolPop(seshat):
    seshat['CC_PolPop'] = seshat['Polity_population'].map(log10float)
    return seshat

# CC_PolTerr 
#   Log_10 of the territory.
def createCC_PolTerr(seshat):
    seshat['CC_PolTerr'] = seshat['Polity_territory'].map(log10float)
    return seshat

# CC_CapPop  
#   Log_10 of the capital's population.
def createCC_CapPop(seshat):
    seshat['CC_CapPop'] = seshat['Population_of_the_largest_settlement']\
        .map(log10float)
    return seshat

# A mean function that treats nans as 0s, but still returns nan if ever
# element is nan.
def hierMean(vals):
    nonNans = [float(v) for v in vals if not np.isnan(float(v))]
    if len(nonNans) == 0:
        return np.nan
    else:
        return np.mean(nonNans)

# CC_Hier    
#   Simply average all non-missing hierarchy variables
def createCC_Hier(seshat):
    seshat['CC_Hier'] = seshat[[
        'Administrative_levels',
        'Military_levels',
        'Religious_levels',
        'Settlement_hierarchy']].apply(hierMean, axis=1)
    return seshat


# CC_Govt    
#   Add together and normalize govt variables.
#   Only include the govt entry if there's at least one non-null variable.
def createCC_Govt(seshat):
    govtVarData = seshat[[
        'Specialized_government_buildings',
        'Professional_lawyers',
        'Professional_military_officers',
        'Professional_priesthood',
        'Professional_soldiers',
        'Bureaucracy_examination_system',
        'Bureaucracy_merit_promotion',
        'Fulltime_bureaucrats',
        'Courts',
        'Formal_legal_code',
        'Judges']]
    # Nix entries with all nulls
    govtVarData = govtVarData[govtVarData.isna().sum(axis=1) < 11]
    seshat['CC_Govt'] = govtVarData.sum(axis=1) / 11.0
    return seshat


# CC_Infra   
#   Sum and normalize infrasturcture variables.
#   Only include the infra entry if there's at least one non-null variable.
def createCC_Infra(seshat):
    infraVarData = seshat[[
        'Bridges',
        'Canals',
        'Ports',
        'Mines_or_quarries',
        'Roads',
        'Irrigation_systems',
        'Markets',
        'Food_storage_sites',
        'Drinking_water_supply_systems']]
    infraVarData = infraVarData[infraVarData.isna().sum(axis=1) < 9]
    seshat['CC_Infra'] = infraVarData.sum(axis=1) / 9.0
    return seshat

#   0 = No evidence (clear absense; not simply null)
#   1 = At least mnemonic devices present
#   2 = At least non-written records present
#   3 = At least script present
#   4 = Written records present
# Presence or absence of a “less sophisticated” writing variable doesn't 
# affect this scale (so if “script” is present, it does not matter wither 
# non-written records are present or absent). (Turchin 2018)
def determineWritingScore(series):
    if series['Written_records'] >= 0.9:
        return 4
    elif series['Script'] >= 0.9:
        return 3
    elif series['Nonwritten_records'] >= 0.9:
        return 2
    elif series['Mnemonic_devices'] >= 0.9:
        return 1
    elif series['Mnemonic_devices'] <= 0.1:
        return 0
    else:
        return np.nan

# CC_Writing 
def createCC_Writing(seshat):
    writingData = seshat[[
        'Mnemonic_devices',
        'Nonwritten_records',
        'Script',
        'Written_records']]
    seshat['CC_Writing'] = writingData.apply(determineWritingScore, axis=1)
    return seshat


def countAbsolutePresents(series):
    count = 0
    for item in series:
        if float(item) == 1.0:
            count += 1
    return count

# CC_Texts   
#   Sum and normalize texts variables.
#   Ignore null entries. Only sum if absolutely "present" (1.0).
#   Ignore "inferred present" (0.9) and below.
#   Only count polities with at data for at least one text variable.
def createCC_Texts(seshat):
    textVarData = seshat[[
        'Lists_tables_and_classifications',
        'Calendar',
        'Sacred_texts',
        'Religious_literature',
        'Practical_literature',
        'History',
        'Philosophy',
        'Scientific_literature',
        'Fiction']]
    textVarData = textVarData[textVarData.isna().sum(axis=1) < 9]
    seshat['CC_Texts'] = textVarData.apply(countAbsolutePresents, axis=1) / 9.0
    return seshat

#   Similar to the writing scale:
#       0 = clear absense
#       1 = Articles
#       2 = Tokens
#       3 = Precious metals
#       4 = Foreign coins
#       5 = Indigenous coins
#       6 = Paper currency
def determineMoneyScore(series):
    if series['Paper_currency'] >= 0.9:
        return 6
    elif series['Indigenous_coins'] >= 0.9:
        return 5
    elif series['Foreign_coins'] >= 0.9:
        return 4
    elif series['Precious_metals'] >= 0.9:
        return 3
    elif series['Tokens'] >= 0.9:
        return 2
    elif series['Articles'] >= 0.9:
        return 1
    elif series['Articles'] <= 0.1:
        return 0
    else:
        return np.nan

# CC_Money
def createCC_Money(seshat):
    moneyVarData = seshat[[
        'Articles',
        'Tokens',
        'Precious_metals',
        'Foreign_coins',
        'Indigenous_coins',
        'Paper_currency']]
    seshat['CC_Money'] = moneyVarData.apply(determineMoneyScore, axis=1)
    return seshat

# Create the "complexity component" variables used for imputation
def createCCs(seshat):
    seshat = createCC_PolPop(seshat)
    seshat = createCC_PolTerr(seshat)
    seshat = createCC_CapPop(seshat)
    seshat = createCC_Hier(seshat)
    seshat = createCC_Govt(seshat)
    seshat = createCC_Infra(seshat)
    seshat = createCC_Writing(seshat)
    seshat = createCC_Texts(seshat)
    seshat = createCC_Money(seshat)
    return seshat

def roundToCentury(year):
    return np.nan if np.isnan(year) else int(np.ceil(year / 100.0)) * 100

def makeTimeSeriesEntry(polityInfo, t):
    polityData = polityInfo[polityInfo['Period_start'] <= t]
    polityData = polityData.sort_values('Period_start')
    series     = polityData.reset_index().iloc[-1].copy()
    series['Temperoculture'] = series['BasePolity'] + ('+' if t >= 0 else '') + str(int(t)) 
    series['Period_start']   = t
    series['Period_end']     = t + 100
    return series

# Create a century-resolved time series for the given polity
def makeTimeSeries(seshat, polity):
    # Get info for just this polity
    polityInfo = seshat[seshat['BasePolity'] == polity]
    # Get the start year
    startYear = roundToCentury(np.min(polityInfo['Period_start']))
    # Get the end year
    endYear   = roundToCentury(np.max(polityInfo['Period_end']))

    if np.isnan(startYear):
        return []

    return [
        makeTimeSeriesEntry(polityInfo.copy(), t)
        for t in range(startYear,endYear+100,100)
    ]

# Turn each polity into a timeseries with century-level resolution
def createFullTimeSeries(seshat):
    # Get the list of all polities
    seshat['BasePolity'] = seshat['Temperoculture'].map(lambda x: x.split('-')[0])
    polities = seshat['BasePolity'].unique()

    # For each polity, generate its timeseries
    serieses = []
    for p in polities:
        series = np.array(makeTimeSeries(seshat,p))
        if series.shape[0] > 0:
            serieses.append(series)

    columns = ['i'] + list(seshat.columns)
    seshat = pd.DataFrame(np.concatenate(serieses), columns=columns)
    seshat = seshat.drop('i',axis='columns')
    return seshat

def export(seshat, timeResolved=True):
    # Remove unnamed columns
    seshat = seshat.loc[:, ~seshat.columns.str.contains('^Unnamed')]
    # Set index
    seshat = seshat.set_index('Temperoculture')
    # Write
    outFile = OUT_FILENAME if timeResolved else OUT_UNRESOLVED_FILENAME
    seshat.to_csv(outFile, sep=',')

def fillInWritingInfo(seshat):
    for i, row in seshat.iterrows():
        score = row['CC_Writing']
        if score >= 4:
            seshat.at[i,'Written_records'] = 0.8
        elif score >= 3:
            seshat.at[i,'Script'] = 0.8
        elif score >= 2:
            seshat.at[i,'Nonwritten_records'] = 0.8
        elif score >= 1:
            seshat.at[i,'Mnemonic_devices'] = 0.8
        elif score < 1:
            seshat.at[i,'Mnemonic_devices'] = 0.2
    return seshat

def fillInMoneyInfo(seshat):
    for i, row in seshat.iterrows():
        score = row['CC_Money']
        if score >= 6:
            seshat.at[i,'Paper_currency'] = 0.8
        elif score >= 5:
            seshat.at[i,'Indigenous_coins'] = 0.8
        elif score >= 4:
            seshat.at[i,'Foreign_coins'] = 0.8
        elif score >= 3:
            seshat.at[i,'Precious_metals'] = 0.8
        elif score >= 2:
            seshat.at[i,'Tokens'] = 0.8
        elif score >= 1:
            seshat.at[i,'Articles'] = 0.8
        elif score < 1:
            seshat.at[i,'Articles'] = 0.2
    return seshat

# Using the imputed CCs, fill in as much data as we can garner from the new CCs
def fillInfoFromImputedCCs(seshat):
    # Log_10 inverse
    log10Inv = lambda x: np.round(10**x)
    # Polity Population
    seshat['Polity_population'] = seshat['CC_PolPop'].map(log10Inv)
    # Polity Territory
    seshat['Polity_territory'] = seshat['CC_PolTerr'].map(log10Inv)
    # Population of the largest settlement
    seshat['Population_of_the_largest_settlement'] = seshat['CC_CapPop'].map(log10Inv)
    # Writing characteristics
    seshat = fillInWritingInfo(seshat)
    # Money characteristics
    seshat = fillInMoneyInfo(seshat)
    return seshat

def main():
#    ensureReqs();                            PROGRESS_BAR.update(1)
#    seshat = getSeshat();                    PROGRESS_BAR.update(1)
#    seshat = phase0Tidy(seshat);             PROGRESS_BAR.update(1)
#    seshat = makeTemperocultureWise(seshat)
    if DEBUG:
        seshat = pd.read_csv('phase1.csv',index_col=0)
    seshat = phase1Tidy(seshat)
    seshat = createCCs(seshat)
    export(seshat, timeResolved=False) 
    seshat = createFullTimeSeries(seshat)
    seshat.to_csv('shiny-seshat-unimputed.csv')
    seshat = firstImpute(seshat)
    seshat = fillInfoFromImputedCCs(seshat)
    seshat = secondImpute(seshat)
    export(seshat)
    print("Exported shiny-seshat.csv!")

if __name__ == '__main__':
    main()
