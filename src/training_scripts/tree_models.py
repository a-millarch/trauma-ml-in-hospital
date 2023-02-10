import os
import sys
from pathlib import Path
from platform import python_version 
import gc
import inspect
from copy import deepcopy
#from imp import reload
import dill

# Custom modules
tr = "" # abs path to workdir
sys.path.append(tr)

import src
from src.models.tree_utils import *
from src.models.ml_utils import precision_recall_from_j_stat, sensivity_specifity_cutoff, resample_to
from src.data.dataloader import import_TQIP_data, import_DK_data, data_version,get_mixed_data, get_random_mixed_data, cat_names, cont_names
from src.visualization.visualize import rf_plot_precision_recall, plot_ranfor_roc, plot_fti

#  model imports
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

import mlflow
from mlflow import MlflowClient

import argparse
import yaml

note = ""

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
    handle =  tr+'src/training_scripts/conf/'+args.exp+'/tree/'+args.config_file
    with open(handle, 'r') as file:
        cfg = yaml.safe_load(file)


    dep_var= defaults["defaults"]["dep_var"]
    
    try:
        resampling = cfg["data"]["resampling"]
    except:
        resampling = None


    #  Load data
    TQIP = import_TQIP_data(tr, dep_var=dep_var)
    dk_complete, dk, dk_test = import_DK_data(tr, dep_var=dep_var)
                               
    if cfg["data"]["train_data_source"] == "DK":
        df = dk
    elif cfg["data"]["train_data_source"] == "US":
        df = TQIP
    elif cfg["data"]["train_data_source"] == "mixed":
        df = get_mixed_data(tr,dk,dep_var)

    elif cfg["data"]["train_data_source"] == "mixed_random":
        df = get_random_mixed_data(tr,dk,dep_var)    
    else:
        print("error setting training dataframe")

    trn_xs, trn_ys, val_xs, val_ys  = create_trn_val(df, cat_names, cont_names, dep_var)    
    print("data loaded\n")

    if resampling is not None:
        trn_xs, trn_ys =resample_to(trn_xs, trn_ys, ss=0.5, kn=3, oversample_method=resampling)
    # Fit func
    run_tags = {"dep_var": dep_var, "dk_data_version":data_version, "note":note}#cfg["run"]["note"]}
    mlflow.set_experiment(defaults["experiment"]["experiment_name"])#cfg["run"]["experiment_name"])

    def get_threshold(df, model,cat_names, cont_names, dep_var):
        xs,ys =proc_split(df, cat_names, cont_names, dep_var)
        preds_pos, ys = rf_test_preds(model,xs, ys ,plot=False)
        if dep_var == "DECEASED":
            threshold = sensivity_specifity_cutoff(ys, preds_pos, method="Fbeta-score")
        elif dep_var == "LOS48HOURSPLUS":
            threshold = sensivity_specifity_cutoff(ys, preds_pos, method="F-score")
        return threshold


    def logged_fit(model,run_name):
        with mlflow.start_run(run_name=run_name+cfg["run"]["run_suffix"], tags= run_tags) as run:
            model.fit(trn_xs, trn_ys)   

            # validation df
            predictions = model.predict_proba(val_xs)#[:,1]
            pred_pos = predictions[:,1]
            # dont rly need
            roc_auc = roc_auc_score(val_ys, pred_pos)      
            val_precision, val_recall = precision_recall_from_j_stat(model, val_xs, val_ys)
            threshold = sensivity_specifity_cutoff(val_ys, pred_pos)

            # test df - get preds, score
            test_xs,test_ys =proc_split(dk_test, cat_names, cont_names, dep_var)
            test_preds_pos, test_ys = rf_test_preds(model,test_xs,test_ys ,plot=True)

            test_auc = roc_auc_score(test_ys, test_preds_pos)
            print("test ROC-AUC\t", test_auc, "\n")
            AP_score = average_precision_score(test_ys, test_preds_pos)
            
            # using threshold from danish dataset predictions (non-testset)
            dk_threshold=get_threshold(dk,model,cat_names, cont_names, dep_var)
            test_precision, test_recall = precision_recall_from_j_stat(model, test_xs, test_ys, threshold= dk_threshold )


            log_metrics = { "valid_roc_auc":roc_auc,
                            "valid_precision_score":val_precision,
                            "valid_recall_score":val_recall,

                            "test_roc_auc":test_auc,
                            "test_precision_score":test_precision,
                            "test_recall_score":test_recall,
                            "test_ap_score": AP_score
                            }
            for name, var in log_metrics.items():
                mlflow.log_metric(name, var)

            log_params = {  "deceased_alt_mode": defaults["defaults"]["dk_data_alt_mode"],
                            "dk_pre_fill_missing": defaults["defaults"]["pre_fill_missing"],
                            "train_data_source":cfg["data"]["train_data_source"],
                          "resampling":resampling
                         }  
            for name, var in log_params.items():
                mlflow.log_param(name, var)
                
            mlflow.log_artifact(handle)

    #  model fit            
    ###################################
    mlflow.sklearn.autolog(silent=True)
    rf = RandomForestClassifier(cfg["hp_rf"]["n_estimators"]
                                , min_samples_leaf=cfg["hp_rf"]["min_samples_leaf"]
                               , class_weight = "balanced"
                               , max_depth = cfg["hp_rf"]["max_depth"])
    logged_fit(rf, run_name="RF")

    ###################################
    mlflow.sklearn.autolog(silent=True)
    AB_model = AdaBoostClassifier(n_estimators=cfg["hp_ab"]["n_estimators"], base_estimator=rf,
                             learning_rate=cfg["hp_ab"]["learning_rate"])
    logged_fit(AB_model,run_name="Adaboost")
    
   
    ###################################
    mlflow.xgboost.autolog(silent=True)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric=cfg["hp_xgb"]["eval_metric"], max_depth = cfg["hp_xgb"]["max_depth"])
    logged_fit(xgb_model, run_name="XGB")
    ###################################
    del(AB_model, rf, xgb_model, TQIP)
    gc.collect()
    ###################################
    mlflow.sklearn.autolog(silent=True)
    ebm = ExplainableBoostingClassifier(random_state=cfg["hp_gam"]["seed"]
                                        , outer_bags = cfg["hp_gam"]["outer_bags"]
                                        , inner_bags=cfg["hp_gam"]["inner_bags"]
                                        , max_leaves = cfg["hp_gam"]["max_leaves"]
                                        , max_bins = cfg["hp_gam"]["max_bins"]
                                      )
    logged_fit(ebm, run_name="EBM")
    # dumping model locally due to non-support in mlflow. Alternatively save as artifact in run
    with open("./"+args.exp+"_ebm_smote_tomek.pkl","wb") as ebm_handle:
        dill.dump(ebm, ebm_handle)
