import os
import sys
import datetime 
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

from azureml.core import Workspace, Datastore, Dataset, Environment
from azureml.data.dataset_factory import DataType

tr ='' #custom path to wd
sys.path.append(tr)
from src.features.utils import *

current_date = str(datetime.now())[0:10].replace('-','')

# loading data in Azure platform
ds = Dataset.Tabular.from_delimited_files(path= tr+'data/raw/tqip/PUF_TRAUMA.csv')
df = ds.to_pandas_dataframe()

TQIP = df.copy(deep=True)
TQIP.dropna(subset=["LOSDays"], inplace=True)
TQIP.reset_index(inplace=True, drop=True)

#  Correct or null faulty registrations of height and weight
TQIP.loc[TQIP['TEMPERATURE'] > 50, "TEMPERATURE"] = np.nan
TQIP.loc[TQIP['TEMPERATURE'] < 15, "TEMPERATURE"] = np.nan

def inches_to_cm(inches): return inches*2.54
def feet_to_cm(feet): return feet*30.48 

TQIP.loc[TQIP["HEIGHT"] < 5, "HEIGHT"] = np.nan
TQIP.loc[TQIP["HEIGHT"] >215, "HEIGHT"] = np.nan

m = TQIP[((TQIP["AGEYEARS"] >16) | (TQIP["AGEYEARS"].isnull()) ) & (TQIP["HEIGHT"] < 120)].copy(deep=True)

m["cm_from_inches"] = m["HEIGHT"].apply( lambda x : inches_to_cm(x))
m["cm_from_feet"] = m["HEIGHT"].apply( lambda x : feet_to_cm(x))

feet = m[m["cm_from_feet"]<215]
m = m.drop(feet.index)

null_m =  m[(m["cm_from_inches"] <120) | (m["cm_from_inches"] >215) ]
not_m = m[(m["cm_from_inches"] >120) & (m["cm_from_inches"] <215)]

TQIP.loc[null_m.index, "HEIGHT"] = np.nan
TQIP.loc[not_m.index, "HEIGHT"] = not_m.loc[not_m.index, "cm_from_inches"]

def pounds_to_kg(pounds): return pounds*0.45359237
def ounces_to_kg(ounces): return ounces*0.0283495231

TQIP.loc[((TQIP["AGEYEARS"] >16) | (TQIP["AGEYEARS"].isnull()))  & (TQIP["WEIGHT"] < 35), "WEIGHT"] = np.nan

s = TQIP[TQIP["WEIGHT"] > 200].copy(deep=True)
s["kg_from_pounds"] = s["WEIGHT"].apply( lambda x : pounds_to_kg(x))
s["kg_from_ounces"] = s["WEIGHT"].apply( lambda x : ounces_to_kg(x))

null_s = s[s["kg_from_pounds"] > 200 ]
not_s = s[s["kg_from_pounds"] < 200 ]


TQIP.loc[null_s.index, "WEIGHT"] = np.nan
TQIP.loc[not_m.index, "WEIGHT"] = not_s.loc[not_s.index, "kg_from_pounds"]

TQIP.loc[TQIP["WEIGHT"] > 200, "WEIGHT"] = np.nan
TQIP.loc[(TQIP["HEIGHT"] <60) & (TQIP["WEIGHT"] > 12), ["HEIGHT","WEIGHT"]] = np.nan
TQIP.loc[(TQIP["HEIGHT"] <80) & (TQIP["WEIGHT"] > 30), ["HEIGHT","WEIGHT"]] = np.nan
TQIP.loc[(TQIP["HEIGHT"] <120) & (TQIP["WEIGHT"] > 80), ["HEIGHT","WEIGHT"]] = np.nan

#  Add columns from other files
with open(tr+'data/interim/cause_code_dict.pkl', 'rb') as f:
    cause_codes = pickle.load(f)

for i in cause_codes.keys():
    print("cause codes: ", i)
    TQIP.loc[TQIP.PRIMARYECODEICD10.isin(cause_codes[i]) ,"CAUSE"] = i
    
proc_ds = Dataset.Tabular.from_delimited_files(path= tr+'data/raw/tqip/PUF_ICDPROCEDURE.csv')
proc = proc_ds.to_pandas_dataframe()

proc_lookup_ds =Dataset.Tabular.from_delimited_files(path= tr+ 'data/raw/tqip/PUF_ICDPROCEDURE_LOOKUP.csv')
proc_lookup = proc_lookup_ds.to_pandas_dataframe()

ais_ds = Dataset.Tabular.from_delimited_files(path= tr+'data/raw/tqip/PUF_AISDIAGNOSIS.csv')
ais_df = ais_ds.to_pandas_dataframe() 

# AIS processing approach and categories https://github.com/alexbonde/TQIP/blob/main/1_Data_preprocessing.ipynb

body_reg_num =  [str(x) for x in range(1,10)]
body_reg = ['HEAD','FACE','NECK','THORAX','ABDOMEN','SPINE','UPPER_EXT','LOWER_EXT','UNSPEC']
ais_df['AISSeverity'] = ais_df['AISSeverity'].astype('Int32').astype(str)
ais_df['AISPREDOT'] = ais_df['AISPREDOT'].astype('Int32').astype(str)

pent_keys = ais_df.loc[ais_df["AISPREDOT"].str[2:4] =="60"].inc_key.unique()
TQIP.loc[TQIP.inc_key.isin(pent_keys),"PENETRATION_INJ"] =1 
TQIP["PENETRATION_INJ"].fillna(0, inplace=True)

for x, y in zip(body_reg_num, body_reg):    
    ais_df.loc[ais_df.loc[(ais_df['AISPREDOT'].str.startswith(x))].index,y] = ais_df.loc[ais_df.loc[
        (ais_df['AISPREDOT'].str.startswith(x))].index,'AISSeverity']

ais_df=ais_df.drop(['AISPREDOT', 'AISPREDOT_BIU', 'AISSeverity', 'AISSeverity_BIU', 'AISVersion'], axis=1)

for i in body_reg:
    globals()[i + '_Inj'] = pd.crosstab(ais_df['inc_key'], ais_df[i]).add_prefix(i + '_')
    
data_frames = HEAD_Inj, FACE_Inj, NECK_Inj, THORAX_Inj, ABDOMEN_Inj, SPINE_Inj, UPPER_EXT_Inj, LOWER_EXT_Inj, UNSPEC_Inj

ais_df = reduce(lambda  left,right: pd.merge(left,right,on=['inc_key'], how='outer'), data_frames)

ais_df = ais_df.fillna(0)

ais_df.reset_index(inplace=True)

TQIP = TQIP.merge(ais_df, on = 'inc_key', how = 'left')

TQIP.columns = TQIP.columns.str.replace("[<,>]", "")

ais_cols = ['HEAD_1', 'HEAD_2', 'HEAD_3', 'HEAD_4', 'HEAD_5', 'HEAD_6',
       'HEAD_9', 'HEAD_NA', 'FACE_1', 'FACE_2', 'FACE_3', 'FACE_4', 'FACE_5',
       'FACE_9', 'FACE_NA', 'NECK_1', 'NECK_2', 'NECK_3', 'NECK_4', 'NECK_5',
       'NECK_6', 'NECK_9', 'THORAX_1', 'THORAX_2', 'THORAX_3', 'THORAX_4',
       'THORAX_5', 'THORAX_6', 'THORAX_9', 'THORAX_NA', 'ABDOMEN_1',
       'ABDOMEN_2', 'ABDOMEN_3', 'ABDOMEN_4', 'ABDOMEN_5', 'ABDOMEN_6',
       'ABDOMEN_9', 'ABDOMEN_NA', 'SPINE_1', 'SPINE_2', 'SPINE_3', 'SPINE_4',
       'SPINE_5', 'SPINE_6', 'SPINE_9', 'SPINE_NA', 'UPPER_EXT_1',
       'UPPER_EXT_2', 'UPPER_EXT_3', 'UPPER_EXT_4', 'UPPER_EXT_5',
       'UPPER_EXT_6', 'UPPER_EXT_9', 'UPPER_EXT_NA', 'LOWER_EXT_1',
       'LOWER_EXT_2', 'LOWER_EXT_3', 'LOWER_EXT_4', 'LOWER_EXT_5',
       'LOWER_EXT_6', 'LOWER_EXT_9', 'LOWER_EXT_NA', 'UNSPEC_1', 'UNSPEC_2',
       'UNSPEC_3', 'UNSPEC_4', 'UNSPEC_5', 'UNSPEC_6', 'UNSPEC_9',
       'UNSPEC_NA']
       
#  Data types
ordinal_columns = ['EMSGCSEYE', 'EMSGCSVERBAL', 'GCSEYE', 'GCSVERBAL', 'GCSMOTOR', 'VERIFICATIONLEVEL', 
                  'PEDIATRICVERIFICATIONLEVEL', 'STATEDESIGNATION', 'STATEPEDIATRICDESIGNATION', 'BEDSIZE']
                  
def ordinal(column, sizes):
    TQIP[column] = TQIP[column].astype('category')
    return TQIP[column].cat.set_categories(sizes, ordered=True, inplace=True)
    
ordinal('EMSGCSEYE', ('Opens eyes spontaneously','Opens eyes in response to verbal stimulation',
                      'Opens eyes in response to painful stimulation', 'No eye movement when assessed'))
ordinal('EMSGCSVERBAL', ('Smiles, oriented to sounds, follows objects, interacts (P) | Oriented (A)',
                      'Cries but is consolable, inappropriate interactions (P) | Confused (A)',
                      'Inconsistently consolable, moaning (P) | Inappropriate words (A)',
                      'Inconsolable, agitated (P) | Incomprehensible sounds (A)', 
                      'No vocal response (P) | No verbal response (A)'))
ordinal('EMSGCSMOTOR', ('Appropriate response to stimulation (P) | Obeys commands (A)', 'Localizing pain', 
                        'Withdrawal from pain', 'Flexion to pain', 'Extension to pain', 'No motor response'))
ordinal('GCSEYE', ('Opens eyes spontaneously','Opens eyes in response to verbal stimulation',
                   'Opens eyes in response to painful stimulation', 'No eye movement when assessed'))
ordinal('GCSVERBAL', ('Smiles, oriented to sounds, follows objects, interacts (P) | Oriented (A)',
                      'Cries but is consolable, inappropriate interactions (P) | Confused (A)',
                      'Inconsistently consolable, moaning (P) | Inappropriate words (A)',
                      'Inconsolable, agitated (P) | Incomprehensible sounds (A)', 
                      'No vocal response (P) | No verbal response (A)'))
ordinal('GCSMOTOR', ('Appropriate response to stimulation (P) | Obeys commands (A)', 'Localizing pain', 'Withdrawal from pain',
                      'Flexion to pain', 'Extension to pain', 'No motor response'))
ordinal('VERIFICATIONLEVEL', ('I - Level I Trauma Center', 'II - Level II Trauma Center', 'III - Level III Trauma Center'))
ordinal('PEDIATRICVERIFICATIONLEVEL', ('I - Level I Pediatric Trauma Center', 'II - Level II Pediatric Trauma Center'))
ordinal('STATEDESIGNATION', ('I', 'II', 'III', 'IV', 'Other', 'Not applicable'))
ordinal('STATEPEDIATRICDESIGNATION', ('I', 'II', 'III', 'IV', 'Other', 'Not Applicable'))
ordinal('BEDSIZE', ('<= 200', '201-400','401-600', '> 600'))

for i in ordinal_columns: 
    globals()[i + '_str_vc'] = TQIP[i].value_counts(dropna=False).to_list()
    globals()[i + '_cat_vc'] = TQIP[i].cat.codes.value_counts(dropna=False).tolist()
    print(i, 'string value counts == category value counts:', globals()[i + '_str_vc'] == globals()[i + '_cat_vc'])
    
def diff_nums(df):
    df["diff_GCS"] = df[df['EMSTOTALGCS'].notnull()]['EMSTOTALGCS'].astype(float)- df[df['TOTALGCS'].notnull()]['TOTALGCS'].astype(float)
    df["diff_SBP"] = df[df['EMSSBP'].notnull()]['EMSSBP'].astype(float)- df[df['SBP'].notnull()]['SBP'].astype(float)
diff_nums(TQIP)    

#  mark outcomes
target = []
TQIP.loc[(TQIP['EDDISCHARGEDISPOSITION'] =="Deceased/expired") 
         | (TQIP['HOSPDISCHARGEDISPOSITION'] == "Deceased/Expired"), "DECEASED"] = 1
target.append("DECEASED")

TQIP.loc[TQIP["LOSDays"] >2, 'LOS48HOURSPLUS'] = 1
target.append('LOS48HOURSPLUS')

TQIP[target] = TQIP[target].fillna(0)
TQIP[target] = TQIP[target].astype(int)

# DTD
#  Load DK data to limit hospital complications based on availability
dk = pd.read_pickle(tr+"data/interim/base_df.pkl")
dk.columns = dk.columns.str.replace("[<,>]", "")

#  mapping cause
dk["KontAars_clean"]  = dk.KontAars.str.replace("ALCC0", "", regex=True)#.replace("ALCC","", regex=True)
dk["KontAars_clean"].replace({"ALCC80": "8", 
                               "ALCC90":"8", 
                               "ALCC70":"8", 
                               "6":"8",
                               "7":"8",
                               #"5":"8",
                               "Ukendt":"8"}, inplace=True)

kontaars_original_map = {"1":"Sygdom og tilstand uden direkte sammenhæng med udefra påført læsion",
                "2":"Ulykke",
                "3":"Voldshandling",
                "4":"Selvmord - selvmordsforsøg",
                "5":"Senfølge",
                "6":"Komplet skaderegistrering foretaget på efterfølgende kontakt",
                "7":"Komplet skaderegistrering foretaget på tidligere kontakt",
                "8":"Andet",
                "9":"Uoplyst"}

kontaars_map = {"1":"other",
                "2":"accident",
                "3":"assault",
                "4":"self_harm",
                "5":"other",
                "6":"other",
                "7":"other",
                "8":"other",
                "9":"other"}


dk["CAUSE"] = dk["KontAars_clean"].replace(kontaars_map)

dk.fillna(np.nan, inplace=True)
dk[ais_cols] =dk[ais_cols].fillna(0)

# mark outcomes
dk.loc[dk["LOSDays"] >=2, 'LOS48HOURSPLUS'] = 1
dk["DødDiff"] =  (dk["Dødsdato"]-dk["BWSTDateTime"]).dt.days
dk.loc[dk["DødDiff"]+1 <= dk["LOSDays"], "DECEASEDwithinLOSDays"] = 1

dk.loc[(dk["DECEASEDwithinLOSDays"] ==1), "DECEASED"] =1

dk[target] = dk[target].fillna(0)

# handle outliers
def fix_GCS(df):
    var = "TOTALGCS"
    mask = (df[var].astype(float)>15)
    df.loc[mask, var]=np.nan
fix_GCS(dk) 
def fill_GCS(df):
    var = "TOTALGCS"
    mask = (df[var].isnull())
    df.loc[mask, var]= df["EMSTOTALGCS"]
    
fill_GCS(dk)
def fix_temperature(df):
    var = "TEMPERATURE"
    mask = (df[var].astype(float)<15)
    df.loc[mask, var]=np.nan
    
fix_temperature(dk)

def fix_pulseoximetry(df):
    var = "PULSEOXIMETRY"
    mask = (df[var].astype(float)<50)
    df.loc[mask, var]=np.nan
    
fix_pulseoximetry(dk)

def fix_iss(df):
    var = "ISS_05"
    mask = (df[var].astype(float)>75)
    df.loc[mask, var]=np.nan
    
fix_iss(dk)

dk.sort_values("BWSTDateTime",inplace=True)
dk.reset_index(drop=True, inplace=True)

def diff_nums(df):
    df["diff_GCS"] = df[df['EMSTOTALGCS'].notnull()]['EMSTOTALGCS'].astype(float)- df[df['TOTALGCS'].notnull()]['TOTALGCS'].astype(float)
    df["diff_SBP"] = df[df['EMSSBP'].notnull()]['EMSSBP'].astype(float)- df[df['SBP'].notnull()]['SBP'].astype(float)
diff_nums(dk)


# intersection
dk_cols = dk.columns.tolist()
def list_filt(lst):
    return [x for x in lst if x in dk_cols]
    
cat_names = list_filt(['SEX', 'ASIAN', 'PACIFICISLANDER', 'RACEOTHER', 'AMERICANINDIAN', 'BLACK', 'WHITE', 'ETHNICITY', 
                   'TRANSPORTMODE', 'PREHOSPITALCARDIACARREST', 'TCCGCSLE13', 'TCCSBPLT30', 'TCC10RR29', 'TCCPEN', 'TCCCHEST', 
                   'TCCLONGBONE', 'TCCCRUSHED', 'TCCAMPUTATION', 'TCCPELVIC', 'TCCSKULLFRACTURE', 'TCCPARALYSIS', 
                   'TEACHINGSTATUS', 'BEDSIZE', 'HOSPITALTYPE', 'VERIFICATIONLEVEL', 'PEDIATRICVERIFICATIONLEVEL', 
                   'STATEDESIGNATION', 'STATEPEDIATRICDESIGNATION', 'EMSGCSEYE', 'EMSGCSVERBAL', 'EMSGCSMOTOR', 
                   'GCSQ_SEDATEDPARALYZED', 
                   'GCSQ_EYEOBSTRUCTION', 'GCSQ_INTUBATED', 'GCSQ_VALID', 'CC_ADHD', 'CC_ALCOHOLISM', 'CC_ANGINAPECTORIS', 
                   'CC_ANTICOAGULANT', 'CC_BLEEDING', 'CC_CHEMO', 'CC_CIRRHOSIS', 'CC_COPD', 'CC_CVA', 'CC_DEMENTIA', 
                   'CC_DIABETES',  'CC_CHF', 'CC_HYPERTENSION', 'CC_MI', 'CC_PAD', 
                   'CC_MENTALPERSONALITY', 'CC_RENAL', 'CC_SMOKING', 'CC_STEROID', 'CC_SUBSTANCEABUSE', 
                    'GCSEYE', 'GCSVERBAL', 'GCSMOTOR', 'CAUSE', 'PENETRATION_INJ'
                      ])



cont_names = list_filt(['AGEYEARS', 'WEIGHT', 'HEIGHT', 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE', 'EMSPULSEOXIMETRY', 'EMSTOTALGCS', 
              'EMSRESPONSEMINS', 'EMSSCENEMINS', 'EMSMINS', 'SBP', 'PULSERATE', 'TEMPERATURE', "TOTALGCS",
              'PULSEOXIMETRY','ISS_05', 
                        
             'RESPIRATORYRATE',
                        "diff_SBP", "diff_GCS"                  
                       ])

df_final = TQIP[cat_names+ais_cols+cont_names+ target].copy(deep=True)

df_final.to_pickle(tr+'data/processed/'+current_date+'_TQIP.pkl')

dk_final = dk[['dtr_index', 'CPR_hash']+cat_names+ais_cols+cont_names+target].copy(deep=True)
#  dtypes as in TQIPD
dk_final= dk_final.astype(df_final[cat_names+ais_cols+cont_names].dtypes.to_dict())

dk_final.to_pickle(tr+'data/processed/'+current_date+'_DK.pkl')
