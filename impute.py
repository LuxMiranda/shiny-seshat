import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm

FEATURES = ['Polity_Population','Polity_territory','Population_of_the_largest_settlement','Administrative_levels','Military_levels','Religious_levels','Settlement_hierarchy','Professional_military_officers','Professional_soldiers','Professional_priesthood','Full-time_bureaucrats','Bureaucracy_examination_system','Bureaucracy_merit_promotion','Specialized_government_buildings','Courts','Formal_legal_code','Judges','Professional_lawyers','Irrigation_systems','Drinking_water_supply_systems','Markets','Food_storage_sites','Roads','Bridges','Canals','Ports','Mines_or_quarries','Couriers','Postal_stations','General_postal_service','Mnemonic_devices','Non-written_records','Written_records','Script','Non-phonetic_writing','Phonetic_alphabetic_writing','Lists_tables_and_classifications','Calendar','Sacred_Texts','Religious_literature','Practical_literature','History','Philosophy','Scientific_literature','Fiction','Articles','Tokens','Precious_metals','Foreign_coins','Indigenous_coins','Paper_currency']

BOOLEANS = ['Professional_military_officers', 'Professional_soldiers', 'Professional_priesthood', 'Full-time_bureaucrats', 'Bureaucracy_examination_system', 'Bureaucracy_merit_promotion', 'Specialized_government_buildings', 'Courts', 'Formal_legal_code', 'Judges', 'Professional_lawyers', 'Irrigation_systems', 'Drinking_water_supply_systems', 'Markets', 'Food_storage_sites', 'Roads', 'Bridges', 'Canals', 'Ports', 'Mines_or_quarries', 'Couriers', 'Postal_stations', 'General_postal_service', 'Mnemonic_devices', 'Non-written_records', 'Written_records', 'Script', 'Non-phonetic_writing', 'Phonetic_alphabetic_writing', 'Lists_tables_and_classifications', 'Calendar', 'Sacred_Texts', 'Religious_literature', 'Practical_literature', 'History', 'Philosophy', 'Scientific_literature', 'Fiction', 'Articles', 'Tokens', 'Precious_metals', 'Foreign_coins', 'Indigenous_coins', 'Paper_currency']

def roundBool(x):
    return 0 if x<0.5 else 1

def roundBools(impSeshat):
    for feat in BOOLEANS:
        impSeshat[feat] = impSeshat[feat].apply(roundBool)
    return impSeshat

def impute(shiny):
    shiny = shiny.reset_index()
    polities  = shiny['Polity']
    names     = shiny['Polity_name']
    eraStarts = shiny['Era_start']
    eraEnds   = shiny['Era_end']
    seshat = shiny.set_index('Polity')
    n_polities, n_features = seshat.shape
    print(n_polities)

    # Drop features outside of the defined 51
    seshat = seshat[FEATURES]

    # Keep track of how many features we'll be imputing
    seshat['Percent_imputed'] = seshat.isnull().sum(axis=1) / n_features

    num_imputations = 20
    print('Performing {} imputations...'.format(num_imputations))

    impSeshat = np.matrix([])

    for i in tqdm(range(num_imputations)):
        imp = IterativeImputer(max_iter=100, min_value=0, random_state=i)
        impIter = imp.fit_transform(seshat)
        if impSeshat.size == 0:
            impSeshat = np.matrix(impIter)
        else:
            impSeshat += impIter

    impSeshat /= num_imputations

    impSeshat = pd.DataFrame(impSeshat, columns=seshat.columns)
    impSeshat['Polity']      = polities
    impSeshat['Polity_name'] = names
    impSeshat['Era_start']   = eraStarts
    impSeshat['Era_end']     = eraEnds
    impSeshat = impSeshat.set_index('Polity')
    impSeshat = roundBools(impSeshat)
    # Replace negative imputed populations with 0 population
    impSeshat['Polity_Population'] = impSeshat['Polity_Population'].apply(lambda x: x if x>0 else 0)

    return impSeshat

if __name__ == '__main__':
    shiny = pd.read_csv('shiny-seshat-unimputed.csv', index_col='Polity')
    shiny = impute(shiny)
    shiny.to_csv('shiny-seshat.csv', sep=',')
