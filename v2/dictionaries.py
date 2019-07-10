# Dictionary of gross polity name replacements.
# We're obviously not limited to the weird character convention of the other
# polity IDs; we just want some semblance of consistency.
POLITY_ID_REPLACEMENTS = {
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
    'Ottoman Yemen' : 'OttomanYemen' 
}

NGA_UTMs = {
    'Big Island Hawaii' : ['4Q','5Q'],
    'Chuuk Islands' : ['56N'],
    'Oro PNG' : ['55L'],
    'Lowland Andes' : ['18M'],
    'Cuzco' : ['18L','19L'],
    'Valley of Oaxaca' : ['14Q'],
    'North Colombia' : ['18P'],
    'Cahokia' : ['15S'],
    'Finger Lakes' : ['18T'],
    'Middle Yellow River Valley' : ['50S'],
    'Kansai' : ['53S'],
    'Southern China Hills' : ['48R'],
    'Cambodian Basin' : ['48P'],
    'Central Java' : ['49M'],
    'Kapuasi Basin' : ['49N'],
    'Kachi Plain' : ['42R'],
    'Deccan' : ['43P'],
    'Garo Hills' : ['46R'],
    'Susiana' : ['39R'],
    'Konya Plain' : ['36S'],
    'Yemeni Coastal Plain' : ['38P'],
    'Sogdiana' : ['42S'],
    'Orkhon Valley' : ['48T'],
    'Lena River Valley' : ['52V'],
    'Latium' : ['33T'],
    'Paris Basin' : ['31U'],
    'Iceland' : ['27W'],
    'Upper Egypt' : ['36R'],
    'Niger Inland Delta' : ['30P','30Q'],
    'Ghanaian Coast' : ['30N']
}

COLUMN_NAME_REMAP = {
    'Original_name' : 'Original_culture_name',
    'Examination_system' : 'Bureaucracy_examination_system',
    'Merit_promotion' : 'Bureaucracy_merit_promotion',
    'Source_of_support' : 'Bureaucracy_source_of_support',
    'Time' : 'Measure_of_time',
    'Area' : 'Measure_of_area',	
    'Geometrical' : 'Geometrical_measurements',
    'Length' : 'Measurement_of_length',
    'Volume' : 'Measurement_of_volume',
    'Weight' : 'Measurement_of_weight',
    'Other_site' : 'Other_polity-owned_site',
    'Cost' : 'Monumental_building_cost',
    'Extent' : 'Monumental_building_extent',
    'Height' : 'Monumental_building_height',
    'Other' : 'Other_information'
}

RITUAL_VARIABLES = [
    'Frequency for the audience',
    'Frequency for the ritual specialist',
    'Frequency per participant',
    'Name'
]

## FORMAT:
## OriginColumn : TargetColumn
## Time to fix some MAD variation in the ritual variable names
COLUMN_MERGE = {
    'Written_records_1' : 'Written_records',
    'Most_widespread_collective_ritual_of_the_official_cult_0_frequency_for_the_audience' : 'Most_widespread_collective_ritual_of_the_official_cult_frequency_for_the_audience',
    'Most_widespread_collective_ritual_of_the_official_cult_0_frequency_for_the_ritual_specialist' : 'Most_widespread_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist',
    'Most_widespread_collective_ritual_of_the_official_cult_0_frequency_per_participant' : 'Most_widespread_collective_ritual_of_the_official_cult_frequency_per_participant',
    'Most_widespread_collective_ritual_of_the_official_cult_0_name' : 'Most_widespread_collective_ritual_of_the_official_cult_name',
    'Largest_scale_collective_ritual_of_the_official_cult_0_frequency_for_the_audience' : 'Largest_scale_collective_ritual_of_the_official_cult_frequency_for_the_audience',
    'Largest_scale_collective_ritual_of_the_official_cult_0_frequency_for_the_ritual_specialist' : 'Largest_scale_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist',
    'Largest_scale_collective_ritual_of_the_official_cult_0_frequency_per_participant' : 'Largest_scale_collective_ritual_of_the_official_cult_frequency_per_participant',
    'Largest_scale_collective_ritual_of_the_official_cult_0_name' : 'Largest_scale_collective_ritual_of_the_official_cult_name',
    'Creator_gods' : 'High_gods_(creator_gods)',
    'Drinking_water_supply_systems_1' : 'Drinking_water_supply_systems',
    'Food_storage_sites_1' : 'Food_storage_sites',
    'Markets_1' : 'Markets',
    'Irrigation_systems_1' : 'Irrigation_systems',
    'Professional_lawyers_1' : 'Professional_lawyers',
    'Linguistic_family_1' : 'Linguistic_family',
    'Most_dysphoric_collective_ritual_frequency_for_the_audience' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_audience',
    'Most_dysphoric_ritual_frequency_for_the_audience' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_audience',
    'Most_dysphoric_collective_ritual_frequency_for_the_ritual_specialist' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist',
    'Most_dysphoric_ritual_frequency_for_the_ritual_specialist' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist',
    'Most_dysphoric_ritual_frequency_per_participant' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_per_participant',
    'Most_dysphoric_collective_ritual_frequency_per_participant' : 'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_per_participant',
    'Most_dysphoric_collective_ritual_name' : 'Most_dysphoric_collective_ritual_of_the_official_cult_name',
    'Most_dysphoric_ritual_name' : 'Most_dysphoric_collective_ritual_of_the_official_cult_name',
    'Most_euphoric_collective_ritual_frequency_for_the_audience' : 'Most_euphoric_collective_ritual_of_the_official_cult_frequency_for_the_audience',
    'Most_euphoric_collective_ritual_frequency_for_the_ritual_specialist' : 'Most_euphoric_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist',
    'Most_euphoric_collective_ritual_frequency_per_participant'           : 'Most_euphoric_collective_ritual_of_the_official_cult_frequency_per_participant',
    'Most_euphoric_collective_ritual_name'                                : 'Most_euphoric_collective_ritual_of_the_official_cult_name',
   'Population_of_the_largest_settlement_1' : 'Population_of_the_largest_settlement',
   'Original_name_1' : 'Original_culture_name',
   'Time_1' : 'Measure_of_time',
   'Philosophy_1' : 'Philosophy',
   'Succeeding_(quasi)_polity' : 'Succeeding_(quasi)polity',
   'Nonphonetic__writing' : 'Nonphonetic_writing',
   'Nonphonetic_alphabetic_writing' : 'Nonphonetic_writing',
   'Non_written_records' : 'Nonwritten_records',
}

RITUAL_VARIABLE_RENAMES = {
'Most_euphoric_collective_ritual_of_the_official_cult_frequency_for_the_audience' :
	'Most_euphoric_ritual_frequency_for_audience',
'Most_euphoric_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist' :
	'Most_euphoric_ritual_frequency_for_ritual_specialist',
'Most_euphoric_collective_ritual_of_the_official_cult_frequency_per_participant' :
	'Most_euphoric_ritual_frequency_per_participant',
'Most_frequent_collective_ritual_of_the_official_cult_frequency_for_the_audience' :
	'Most_frequent_ritual_frequency_for_audience',
'Most_frequent_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist' :
	'Most_frequent_ritual_frequency_for_ritual_specialist',
'Most_frequent_collective_ritual_of_the_official_cult_frequency_per_participant' :
	'Most_frequent_ritual_frequency_per_participant',
'Most_frequent_collective_ritual_of_the_official_cult_name' :
	'Most_frequent_ritual_name',
'Most_widespread_collective_ritual_of_the_official_cult_frequency_for_the_audience' :
	'Most_widespread_ritual_frequency_for_audience',
'Most_widespread_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist' :
	'Most_widespread_ritual_frequency_for_ritual_specialist',
'Most_widespread_collective_ritual_of_the_official_cult_frequency_per_participant' :
	'Most_widespread_ritual_frequency_per_participant',
'Most_euphoric_collective_ritual_of_the_official_cult_name' :
	'Most_euphoric_ritual_name',
'Most_widespread_collective_ritual_of_the_official_cult_name' :
	'Most_widespread_ritual_name',
'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_audience' :
	'Most_dysphoric_ritual_frequency_for_audience',
'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist' :
	'Most_dysphoric_ritual_frequency_for_ritual_specialist',
'Most_dysphoric_collective_ritual_of_the_official_cult_frequency_per_participant' :
	'Most_dysphoric_ritual_frequency_per_participant',
'Most_dysphoric_collective_ritual_of_the_official_cult_name' :
	'Most_dysphoric_ritual_name',
'Largest_scale_collective_ritual_of_the_official_cult_frequency_for_the_ritual_specialist':
	'Largest_scale_ritual_frequency_for_ritual_specialist',
'Largest_scale_collective_ritual_of_the_official_cult_frequency_per_participant':
	'Largest_scale_ritual_frequency_per_participant',
'Largest_scale_collective_ritual_of_the_official_cult_name' :
        'Largest_scale_ritual_name',


}

## TODO TrClcER-1 has most euphoric ritual feast