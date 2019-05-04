import pandas as pd
import numpy as np
import urllib.request
import os
import os.path
import math
import io
import re
from tqdm import tqdm
import scrape.get
import scrape.parse

# Munging notes:
# If one variable has multiple values over a range of dates, only the latest-date
# version of the value is stored.
# If a value is a range, the midpoint is stored.
# Dates are stored as signed integers with 1AD centered at 0

SESHAT_URL = 'http://seshatdatabank.info/data/dumps/seshat-20180402.csv'

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
    source      = io.open(file, 'r', encoding=sourceEncoding)
    uncodedData = source.read()
    target      = io.open(encodedName, 'w', encoding=targetEncoding)
    target.write(uncodedData)
    # Remove redundant file
    os.remove(unencodedName)


# Check if a directory is empty, excluding
# hidden files like .gitignore
def directoryEmpty(dr):
    return [f for f in os.listdir(dr) if not f.startswith('.')] == []


# Download all required files
def downloadReqs():
    # Download the database if we haven't already
    if not os.path.isfile('seshat.csv'):
        downloadSeshat()
    if not os.path.isfile('scrape/nga-list.html'):
        scrape.get.downloadNGAlist(basedir='scrape/')
    if directoryEmpty('scrape/ngas/'):
        scrape.get.getPolities(basedir='scrape/')

downloadReqs()

if not os.path.isfile('scrape/nameDates.csv'):
    scrape.parse.parsePolities(basedir='scrape/')

# Read in csv
seshat = pd.read_csv('seshat.csv')

# Rename column titles to work with pandas
seshat = seshat.rename(columns={
    'Value From' : 'Value_From',
    'Value To'   : 'Value_To',
    'Date From'  : 'Date_From',
    'Date To'    : 'Date_To',
    'Fact Type'  : 'Fact_Type',
    'Value Note' : 'Value_Note',
    'Date Note'  : 'Date_Note'
    })

# Remove information on contributors
# responsible for data validation. Sorry, contributors!
seshat = seshat[seshat.Variable != 'RA']
seshat = seshat[seshat.Variable != 'Editor']
seshat = seshat[seshat.Variable != 'Expert']

# Function mapped to Value_From to convert from
# string data to integer data
def valConvert(value):
    value = value.lower()
    nan = float('NaN')
    if value == 'present' or value == 'inferred present' or value == 'inferred inferred present':
        return 1
    elif value == 'absent' or value == 'inferred absent':
        return 0
    elif value in ['unknown','suspected unknown', 'inferred', 'uncoded']:
        return nan
    else:
        return value

# Convert "present" and "absent" variables to boolean integers
seshat['Value_From'] = seshat['Value_From'].apply(valConvert)

# Drop unnecessary columns
seshat = seshat.drop(columns=['Section','Subsection','Fact_Type','Comment'])

# Grab the first polity and NGA in the dataset
curPolity = seshat.iloc[0]['Polity']
curNGA = seshat.iloc[0]['NGA']

# Set up some things
polities = pd.DataFrame()
curSeries = pd.Series()
curSeries['Polity'] = curPolity
curSeries['NGA'] = curNGA

print('Munging polities...')
pbar = tqdm(total=seshat.shape[0])

# Just... Rearrange everything
for index, row in seshat.iterrows():
    pbar.update(1)
    # Check if new row is a new polity
    if row['Polity'] != curPolity:
        #print('Munging polity ' + curPolity + '...')
        # Add the current series to the polities dataframe
        polities = polities.append(curSeries.copy(), ignore_index=True)
        # Reset the current series
        curSeries = pd.Series()
        # Update the new polity
        curPolity = row['Polity']
        curSeries['Polity'] = curPolity
        curSeries['NGA'] = row['NGA']

    # Grab the variable that is actually being expressed
    variable = row['Variable']

    if row['Value_Note'] == 'simple':
        # For simple variables, just store it in the series
        curSeries[variable] = row['Value_From']
    elif row['Value_Note'] == 'range':
        # For ranges, store the midpoint of the high and the low
        curSeries[variable] = (float(row['Value_From']) + float(row['Value_To']))/2.0

pbar.close()

# Delete duplicate columns
polities = polities.drop(columns=[
    'Written records_1',
    'Philosophy_1',
    'Time_1',
    'Population of the largest settlement_1'
    ])

# Standard max function takes NaN as larger than everything
def notDumbMax(l, r):
    if math.isnan(l):
        return r
    elif math.isnan(r):
        return l
    else:
        return max(l,r)

# Convenience function for combining two series via maximum
def combineWithMax(df, s1, s2):
    return df[s1].combine(df[s2], notDumbMax)

# Merge columns that are really the same but have minor spelling differences.
# Columns with capitalized first letters are favored for consistency. 

polities['Drinking water supply systems'] = combineWithMax(
        polities, 
        'Drinking water supply systems', 
        'drinking water supply systems'
        )

polities['Food storage sites'] = combineWithMax(
        polities, 
        'Food storage sites', 
        'food storage sites'
        )

polities['Markets'] = combineWithMax(
        polities, 
        'Markets', 
        'markets'
        )

polities['Nonwritten records'] = combineWithMax(
        polities, 
        'Nonwritten records', 
        'Non written records'
        )

polities['Irrigation systems'] = combineWithMax(
        polities, 
        'Irrigation systems', 
        'irrigation systems'
        )


polities['Professional lawyers'] = combineWithMax(
        polities, 
        'Professional lawyers', 
        'Professional Lawyers'
        )

polities['Non-phonetic writing'] = combineWithMax(
        polities, 
        'Non-phonetic writing', 
        'Non-phonetic  writing'
        )

polities['Non-phonetic writing'] = combineWithMax(
        polities, 
        'Non-phonetic writing', 
        'Non-phonetic alphabetic writing'
        )

# Remove the extra columns that we merged into their more-handsome siblings
polities = polities.drop(columns=[
        'Professional Lawyers',
        'irrigation systems',
        'Non written records',
        'markets',
        'food storage sites',
        'Non-phonetic  writing',
        'Non-phonetic alphabetic writing',
        'drinking water supply systems'
    ])


# Do some re-naming to achieve greater consistency with other columns
# and clarity in the absence of sub-sections
polities = polities.rename(columns={
    'Nonwritten records' : 'Non-written records',
    'cost'               : 'Polity-owned building cost',
    'extent'             : 'Polity-owned building extent',
    'height'             : 'Polity-owned building height',
    'Source of support'  : 'Bureaucracy source of support',
    'Examination system' : 'Bureaucracy examination system',
    'Merit promotion'    : 'Bureaucracy merit promotion',
    'Length'             : 'Measurement of length',
    'Area'               : 'Measurement of Area',
    'Volume'             : 'Measurement of Volume',
    'Weight'             : 'Measurement of Weight',
    'Time'               : 'Measurement of Time',
    'Geometrical'        : 'Measurement of Geometry'
    })

# Re-sort the columns by name
polities = polities.reindex(sorted(polities.columns), axis=1)

# Index by Polity
polities = polities.set_index('Polity')

## Fix some weirdness
polities.at['JpKemmu', 'Store of wealth'] = 1 # This is set to "warehouse" instead of being a simple boolean
polities.at['IdKahur', 'Polity territory'] = 175000 # This still had 'km^2' units left on it

# Drop some redundant entries
polities = polities.drop(['Pre-colonial Garo Hills'
    ,'British colonial period and early independent India'
    ,'Iceland Commonwealth Period (930-1262 CE)'
    ,'Norway Kingdom'
    ,'Ottoman Yemen'
    ,'Modern Yemen'
    ,'Pre-Brooke Raj Period'
    ,'Brooke Raj and Colonial Period'
    ,'Late Qing'
    ,'Early Chinese'
    ,'Pre-Colonial Finger Lakes'
    ,'Iroquois Early Colonial'
    ,'Colonial Lowland Andes'
    ,'Ecuadorian'
    ,'Russia Pre-Russian period'
    ,'Russia Early Russian'
    ,'Oro Pre-Colonial'
    ,'Oro Early Colonial'])


# Drop some ghost entities not listed on the Seshat website.
# They're especially sparse entries, anyway.
polities = polities.drop(['Mali Kingdom of Gao Za Dynasty (700-1080 CE)'
    ,'Mali Kingdom of Gao (1080-1236 CE)'
    ,'Peru Cuzco chiefdom Middle Horizon (650-1000 CE)'
    ,'Peru Lucre Basin (1000-1250 CE)'
    ,'Peru Cuzco Valley Killke (1000-1250)'
    ,'Peru Lucre Basin (1300-1400 CE)'
    ,'Peru Cuzco Valley Killke (1250-1400)'
    ,'MlToucl'])

# Fetch era dates and full polity names
nameDates = pd.read_csv('scrape/nameDates.csv')
# Drop duplicates from the nameDate scraper
nameDates = nameDates.drop_duplicates(subset='Polity')
# Combine our shiny dataset with era ranges and full polity names
polities  = pd.merge(polities, nameDates, on='Polity')

# Delete a weird entry
polities = polities.drop(polities[
    (polities['Polity'] == 'FrBurbL') & (polities['NGA'] == 'Cahokia')].index)

# Group NGAs into lists and clean things up
goodNgas = pd.DataFrame(polities.groupby('Polity').NGA.agg(lambda x: list(x)))
polities = pd.merge(goodNgas,polities,on='Polity')
polities = polities.drop('NGA_y',axis=1)
polities = polities.drop_duplicates(subset='Polity')
polities = polities.set_index('Polity')
polities = polities.rename(columns={'NGA_x': 'NGA'})

# Drop unneccesary columns
polities = polities.drop('Other', axis=1)
polities = polities.drop('Other site', axis=1)

# Check if a string is a date
def isDate(s):
    return (s[-2:] == 'CE')

# Convert a date string to an integer
def dateToInt(date):
    if date[-3:] == 'BCE':
        return -1*int(date[:-3])
    elif date[-2:] == 'CE':
        return int(date[:-2]) - 1
    else:
        raise Exception('Invalid date')

# Convert an era string to a tuple of year integers
def getDatesFromEra(eraStr):
    [start, end] = eraStr.replace(' ','').split('-')
    end = dateToInt(end)
    if isDate(start):
        start = dateToInt(start)
    else:
        start = int(start)*np.sign(end)
        if start > 0:
            start -= 1
    return start,end

# Convert eras to date integers
polities['Era'] = polities['Era'].apply(getDatesFromEra)
polities[['Era_start', 'Era_end']] = pd.DataFrame(polities['Era'].tolist(), index=polities.index)  
polities = polities.drop('Era',axis=1)

# Reorganize columns for human-readability
firstCols = ['NGA','Polity_name','Era_start','Era_end']
cols = list(polities.drop(firstCols,axis=1).columns.values)
polities = polities[firstCols + cols]

# Consistency change
polities = polities.rename(columns={'Polity_Population': 'Polity_population'})

# Fix all columns to use underscores instead of spaces
polities.columns = polities.columns.str.replace(' ', '_')

# Export
polities.to_csv('shiny-seshat.csv', sep=',')
print("Exported to shiny-seshat.csv!")
