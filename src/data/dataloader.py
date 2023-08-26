import pandas as pd
import numpy as np
from src.features.loader import *
data_version = "20230117"


def import_TQIP_data(tr, dep_var="DECEASED"):
    TQIP= pd.read_pickle(tr+'data/processed/'+data_version+'_TQIP.pkl')
    TQIP = TQIP[cat_names+ cont_names+[dep_var,]]
    return TQIP


def import_DK_data(tr, dep_var="DECEASED", alt_mode=False, fill_true_missing=False, seed=42):
    dk_complete = pd.read_pickle(tr+'data/processed/'+data_version+'_DK.pkl')   
    dk_complete = dk_complete[dk_complete.PENETRATION_INJ.notnull()]
    dk_complete = dk_complete[cat_names+ cont_names+[dep_var,]]
    
    dk_test = dk_complete.sample(frac=0.2, random_state=seed).copy(deep=True)
    dk = dk_complete.drop(dk_test.index).copy(deep=True)
    print("length of dk_complete:", len(dk_complete))
    return dk_complete, dk, dk_test


# TL
def check_num_similarity(df_large, df_small):
    for col in cont_names:
        df_small_min= df_small[col].min()
        df_small_max= df_small[col].max()
        
        df_large_min = df_large[col].min()
        df_large_max = df_large[col].max()
        
        if df_small_min < df_large_min: 
            print(col, "min smaller in df_small:\n \tChange", df_small_min,"to" , df_large_min, "for N=",
                  len(df_small[df_small[col] < df_large_min]))
        else: pass#print(col, "min within bounds")
        
        if df_small_max > df_large_max:
            print(col, "max larger in df_small:\n \tChange", df_small_max,"to" , df_large_max, "for N=",
                  len(df_small[df_small[col] > df_large_max]) )

        else: pass#print(col, "max within bounds")
    

def fix_num_similarity(df_large, df_small):
    for col in cont_names:
        df_small_min= df_small[col].min()
        df_small_max= df_small[col].max()
        
        df_large_min = df_large[col].min()
        df_large_max = df_large[col].max()
  
        if df_small_min < df_large_min: 
            df_small.loc[df_small[col] < df_large_min , col] = np.nan
        else: pass

        if df_small_max > df_large_max: 
            df_small.loc[df_small[col] > df_large_max , col] = np.nan
        else: pass

def check_cats_sim(df_large, df_small):

    # Check for discrepancy between categorical classes
    for i in cat_names:
        df_small_lst = df_small[i].unique().tolist()
        df_large_lst = df_large[i].unique().tolist()
        not_df_small = [j for j in df_large_lst if j not in df_small_lst and str(j) !="nan"]
        not_df_large =[j for j in df_small_lst if j not in df_large_lst and str(j) !="nan"]
        #if not_df_small or not_df_large:
        #    print(i, "not in df_small:",not_df_small, "not in df_large:", not_df_large, "\n")
        if not_df_large:
            print(i, "in small df, but not in df_large:", not_df_large, "\n")

def get_transfer_learn_data(data_dict, TQIP, dep_var="DECEASED" ,mode = "keep"):
    """ data_dict: dk_complete, dk_test, dk"""
    
    if mode == "temporal_fixed":
        dk_tl=data_dict["dk_complete"].copy(deep=True)
        #  keep 15 % of all data for test df
        dk_tl_test = dk_tl.iloc[-int((len(dk_tl)*0.2)):].copy(deep=True)
        dk_tl_train_valid = dk_tl.drop(dk_tl_test.index).copy(deep=True)
        dk_tl_test.reset_index(inplace=True, drop=True)
        #  append row with nulls for valid df (transfer learning purposes)
        dk_tl_train_valid = dk_tl_train_valid.append([{"SEX":"Male"}], ignore_index=True)
        #  select indices for validation set. consider choosing at random instead of tail.  
        valid_idx = dk_tl_train_valid.iloc[-int((len(dk_tl_train_valid)*0.2)):].index


    elif mode == "keep":
        dk_tl_test = data_dict["dk_test"].copy(deep=True)
        dk_tl_train_valid = data_dict["dk"].copy(deep=True)
        #dk_tl_test.reset_index(inplace=True, drop=True)
        #  append row with nulls for valid df (transfer learning purposes)
        dk_tl_train_valid = dk_tl_train_valid.append([{"SEX":"Male"}], ignore_index=True)
        #  select indices for validation set.  
        valid_idx = dk_tl_train_valid.sample(frac=0.2).index


    elif mode == 'new_random':
        dk_tl=data_dict["dk_complete"].copy(deep=True)
        #  keep 15 % of all data for test df
        dk_tl_test = dk_tl.sample(frac=0.15).copy(deep=True)
        dk_tl_train_valid = dk_tl.drop(dk_tl_test.index).copy(deep=True)
        dk_tl_test.reset_index(inplace=True, drop=True)
        #  append row with nulls for valid df (transfer learning purposes)
        dk_tl_train_valid = dk_tl_train_valid.append([{"SEX":"Male"}], ignore_index=True)
        #  select indices for validation set. consider choosing at random instead of tail.  
        valid_idx = dk_tl_train_valid.sample(frac=0.2).index
    else: pass

    #  append another null row for training set
    dk_tl_train_valid = dk_tl_train_valid.append([{"SEX":"Male"}], ignore_index=True)
    #  fill target nulls. TO processes will not fillna for dependent variables 
    dk_tl_train_valid[dep_var] = dk_tl_train_valid[dep_var].fillna(0)
    dk_tl_train_valid.reset_index(inplace=True, drop=True)    
    
    check_num_similarity(TQIP, dk_tl_train_valid)   
    fix_num_similarity(TQIP, dk_tl_train_valid)
    check_cats_sim(TQIP, dk_tl_train_valid)    
    
    return dk_tl_test, dk_tl_train_valid, valid_idx


def get_mixed_data(tr,dk, dep_var):
    tqip_merge =pd.read_pickle(tr+'data/processed/'+"20230109"+'_'+dep_var+'_TQIP_merge.pkl')
    tqip_merge = tqip_merge[cat_names+ cont_names+[dep_var,]]
    train_df = pd.concat([dk,tqip_merge])
    return train_df

def get_random_mixed_data(tr,dk, dep_var):
    tqip_merge = import_TQIP_data(tr, dep_var)
    tqip_merge = tqip_merge.sample(n = 50000, random_state=42)
    tqip_merge = tqip_merge[cat_names+ cont_names+[dep_var,]]
    train_df = pd.concat([dk,tqip_merge])
    return train_df
