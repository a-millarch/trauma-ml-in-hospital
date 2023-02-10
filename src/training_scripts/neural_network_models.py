import os
import sys
from pathlib import Path
from platform import python_version 
import gc
import inspect
from copy import deepcopy
import dill

# Custom modules
tr = '' # absolute path to upper working dir

import src
from src.models.fastbinary import *
from src.visualization.visualize import multi_label_roc_plot
from src.data.dataloader import import_TQIP_data, import_DK_data, data_version,get_mixed_data,get_random_mixed_data

import mlflow
from mlflow import MlflowClient

import yaml
import argparse

experiment_name = "trauma_mortality"
note= ""

if __name__=="__main__":
    #  Load config files
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", help="working experiment directory for conf files")
    parser.add_argument("config_file", help="name of config file (def.yaml in /conf/OUTCOME")
    args = parser.parse_args()
  # load defaults
    def_handle = tr+'src/training_scripts/conf/'+args.exp+'/defaults.yaml'
    with open(def_handle, 'r') as file:
        defaults = yaml.safe_load(file)
   # load experiment config files 
    handle =  tr+'src/training_scripts/conf/'+args.exp+'/nn/'+args.config_file
    with open(handle, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Set as local vars for convenience
    dep_var= defaults["defaults"]["dep_var"]
    hyper_parameter_dict = cfg["hpd"]
    weights = cfg["hpd"]["weights"]
    
    # resampling will not be in .yaml if not relevant
    try:
        resampling_baseline = cfg["resampling"]["baseline_method"]
    except:
        resampling_baseline = None   
    try:
        resampling_tl = cfg["resampling"]["tl_method"]
    except:
        resampling_tl = None

    #  Load base datasets
    TQIP = import_TQIP_data(tr, dep_var=dep_var)
    dk_complete, dk, dk_test = import_DK_data(tr, dep_var=dep_var )
    # selecting dataset for training non-TL model
    if cfg["data"]["train_data_source"] == "mixed":
        df =get_mixed_data(tr, dk, dep_var)
        
    elif cfg["data"]["train_data_source"] == "mixed_random":
        df =get_random_mixed_data(tr, dk, dep_var)
        
    elif cfg["data"]["train_data_source"] == "US":
        df =TQIP
    else:
        df=dk.copy(deep=True)      
    print("data loaded")

    # 
    run_tags = {"dep_var": dep_var, "dk_data_version":data_version, "note":note}#cfg["run"]["note"]}
    mlflow.set_experiment(defaults["experiment"]["experiment_name"])

    def logged_fit(model, test_df, n_epochs=5,  run_name=str(datetime.now())):

        mlflow.fastai.autolog()

        with mlflow.start_run(run_name=run_name+cfg["run"]["run_suffix"], tags= run_tags) as run:
            model.fit(n_epochs)
            model.get_test_preds(test_df,"dk_test", threshold_df = dk)

            log_metrics = { "test_roc_auc":model.test_roc_auc[dep_var],
                            "test_precision_score":model.test_precision,
                            "test_recall_score":model.test_recall,
                            "test_ap_score": model.AP_score
                            }
            for name, var in log_metrics.items():
                mlflow.log_metric(name, var)
                
            log_params = {   "layer_sizes":model.hpd["nn_size"]
                            ,"hpd":model.hpd
                            ,"deceased_alt_mode": defaults["defaults"]["dk_data_alt_mode"]
                            ,"dk_pre_fill_missing": defaults["defaults"]["pre_fill_missing"]
                            ,"class_weights": model.weights
                            ,"resampling": model.resampling
                            ,"lr_val": model.lr_val
                            , "train_data_source":cfg["data"]["train_data_source"]
                          }
            for name, var in log_params.items():
                mlflow.log_param(name, var)
            # log roc-auc, pr plot saved at tmp. location    
            mlflow.log_artifact("./mlflow_figs/metric_plot.png")
            # log .yaml file
            mlflow.log_artifact(handle)
            
        mlflow.fastai.autolog(disable=True)    

    #  Model creation
    baseline = BinarySingleLabelNN(df, target=dep_var, test_df = dk_test
                                  ,default_setup=True, hyper_parameter_dict = hyper_parameter_dict
                                   ,resampling = resampling_baseline
                                   , weights = weights)
                            
    logged_fit(baseline,dk_test,cfg["hpd"]["n_epochs"],run_name="DK")   
    
    #  Transfer learning
    from src.data.dataloader import get_transfer_learn_data

    dk_data_dict = {"dk_complete":dk_complete, 
                    "dk":dk, 
                    "dk_test":dk_test}          

    dk_tl_test, dk_tl_train_valid, valid_idx = get_transfer_learn_data(dk_data_dict, TQIP, dep_var=dep_var)

    if cfg["tl"]["US_data"] == "rf_mixed":
        # using train dk - will be data to retrain on as well
        US_data =get_mixed_data(tr, dk, dep_var)

    elif cfg["tl"]["US_data"] == "rf_mixed_random":
        # using train dk - will be data to retrain on as well
        US_data =get_random_mixed_data(tr, dk, dep_var)    
    else:
        US_data = TQIP

    #  US model handling
    if cfg["tl"]["load"]:
        try: 
            with open(tr+"models/tqip_tl_"+dep_var+"_"+data_version+".pkl", 'rb') as f:
                tqip_tl = pickle.load(f)
        except: 
            print("using 14th dec version")
            with open(tr+"models/tqip_tl20221214.pkl", 'rb') as f:
                tqip_tl = pickle.load(f)
    else: 
        #tqip_hyper_parameter_dict = {"nn_size" : [1000,500], "patience":3, "batch_size":1024}
        hyper_parameter_dict["cb_patience"] = 2
        hyper_parameter_dict["batch_size"] = 1024
        tqip_tl =BinarySingleLabelNN(US_data, default_setup=True 
                                           , target= dep_var,  hyper_parameter_dict = hyper_parameter_dict
                                      , weights=weights
                                    , resampling = resampling_tl) 
        tqip_tl.fit(5)

        if cfg["tl"]["write"]:
            with open(tr+"models/tqip_tl_"+dep_var+"_"+data_version+".pkl", 'wb') as f:
                    pickle.dump(tqip_tl, f)

    if cfg["tl"]["freeze"]:
        for param in tqip_tl.learn.layers[1:4].parameters():
            param.requires_grad = False


    if cfg["tl"]["dk_dls"]:
        hyper_parameter_dict["cb_patience"] = 5
        hyper_parameter_dict["batch_size"] = 64

        dk_tl =BinarySingleLabelNN(dk_tl_train_valid, default_setup=False , valid_idx = valid_idx.copy()
                                           , target= dep_var, hyper_parameter_dict = hyper_parameter_dict
                                   ,resampling = resampling_baseline
                                    #, weights=True
                                    )
        dk_tl.get_to(split_method='idx')

        tqip_tl.learn.dls = dk_tl.to

        tqip_tl.hpd["patience"] = 5
        tqip_tl.hpd["batch_size"] = 64

    else:
        dk_tl_valid = dk_tl_train_valid.iloc[valid_idx]
        dk_tl_train = dk_tl_train_valid.drop(dk_tl_valid.index)

        to_tst =tqip_tl.to.new(dk_tl_train)
        to_tst.process()

        to_tst_val = tqip_tl.to.new(dk_tl_valid)
        to_tst_val.process()

        t_trn_dl = TabDataLoader(to_tst.train,  bs=64)
        t_val_dl = TabDataLoader(to_tst_val, bs=64)
        tst_dls = DataLoaders(t_trn_dl, t_val_dl)

        tqip_tl.learn.dls = tst_dls 

    tqip_tl.lr_val = tqip_tl.lr_val*cfg["tl"]["tl_lr_factor"]

    logged_fit(tqip_tl, dk_tl_test,cfg["hpd"]["n_epochs"], run_name = "TL")
