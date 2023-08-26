import os
import sys

import fastai
from fastai.tabular.all import *

from src.features.loader import target, cont_names, cat_names, ais_names, col_selector, resampling_dict
from src.models.ml_utils import * 
from src.visualization.visualize import multi_label_roc_plot

from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss, precision_recall_curve, f1_score, average_precision_score

import mlflow.fastai
from mlflow import MlflowClient

class BinarySingleLabelNN():
    def __init__(self, df
                 , cat_names = cat_names , cont_names = cont_names
                 , target = target
                 , test_df=None
                 , hyper_parameter_dict = None
                 , seed=42, resampling=None, resampling_dict = resampling_dict
                 , default_setup = True, valid_idx=None
                , weights=False):

        self.cat_names = list(cat_names)
        self.cont_names = list(cont_names)

        self.seed = int(seed)
        self.default_setup=default_setup
        self.resampling = resampling
        self.weights  = weights  
        self.hpd = dict(hyper_parameter_dict)
        self.valid_idx = valid_idx   

        # CONSIDER THIS!
        if isinstance(target, str):
            self.target= [target,]
        else:
            self.target = target

        self.df = df[cat_names+cont_names+self.target].copy(deep=True)
        
      
        self.test_df = test_df


        self.to = None
        self.to_valid = None
        self.dls = None
        self.test_dataframes_dict = dict()
        self.test_results = pd.DataFrame()
        self.resampling_dict = resampling_dict

        if self.default_setup:
            if self.resampling and self.target[0] in self.resampling_dict.keys():
                self.resample_data()
                self.add_standard_learner()
            else:
                self.get_to() 
                self.get_dls()
                self.add_standard_learner()
        else:
            pass

    def get_to(self, df = None, split_method="random"):
        cat = self.cat_names.copy()
    #    ais = self.ais_names.copy()
        cont = self.cont_names.copy() 

        if df is None:
            df = self.df

        procs_nn = [Categorify, FillMissing(fill_strategy = FillStrategy.mode) , Normalize]
        y_block = CategoryBlock()

        if split_method == 'idx':
            print("splitting by idx")
            to = TabularDataLoaders.from_df(df
                                , procs=procs_nn
                                , cat_names=cat
                                , cont_names=cont
                                , y_names=self.target
                                , y_block=y_block
                                , split_idx=self.valid_idx
                              )

        elif split_method == 'random' :
            splits = RandomSplitter(seed = self.seed, valid_pct=0.2)(range_of(df))
            to = TabularPandas(df
                                , procs=procs_nn
                                , cat_names=cat
                                , cont_names=cont
                                , y_names=self.target
                                , y_block=y_block
                                , splits=splits)
        else:
                to = TabularPandas(df
                                    , procs=procs_nn
                                    , cat_names=cat
                                    , cont_names=cont
                                    , y_names=self.target
                                    , y_block=y_block)

        self.to = to


    def get_dls(self):
        if not self.to:
            print("No to, getting")
            self.get_to()
        else:
            pass

        dls = self.to.dataloaders()
       # dls.rng.seed(42)
        #shuffled_dataset = torch.utils.data.Subset(self.to.train, torch.randperm(len(my_dataset)).tolist()
        try:
            trn_dl = TabDataLoader(self.to.train, shuffle=True, bs=self.hpd ["batch_size"], drop_last=True)
        except:
            trn_dl = TabDataLoader(self.to.train,  bs=self.hpd ["batch_size"], drop_last=True)

        val_dl = TabDataLoader(self.to.valid, bs=self.hpd ["batch_size"])
        self.dls = DataLoaders(trn_dl, val_dl)
        

    def add_standard_learner(self):   
        """
        Create FastAI neural network leaner with "standard" configuration
        """
        if not self.dls:
            self.get_dls()

        if isinstance(self.weights, bool):
            if self.weights:
                print("calculating weights for weighted loss func")
                self.weights = get_class_weights(self.df, self.target)
                loss_func = CrossEntropyLossFlat(weight=self.weights)
            
            else:
                loss_func = CrossEntropyLossFlat()

        elif isinstance(self.weights, list):
            self.weights = torch.FloatTensor(self.weights)
            loss_func = CrossEntropyLossFlat(weight=self.weights)
        else:
            print("invalid weights input, no weighting")
            self.weights = False
            loss_func = CrossEntropyLossFlat()
            #loss_func = FocalLossFlat()

        learn = tabular_learner(self.dls, layers =self.hpd ["nn_size"],
                                metrics=[RocAucBinary(), F1Score(), FBeta(beta=2), Precision(),  Recall()], 
                                loss_func = loss_func,
                                config=tabular_config(ps=self.hpd["ps"]
                                                      , embed_p=self.hpd["embed_p"] 
                                                      ,use_bn=True)
                               )

        lr_slide, lr_valley = learn.lr_find(suggest_funcs=(slide, valley), show_plot=False)
        lr_val = (lr_slide+ lr_valley)*self.hpd["lr_val_factor"]

        self.keep_path = learn.path
        learn.path = Path('./models')

        self.lr_val = lr_val
        self.learn = learn
        self.learn.pile = "slp"
        return self 
    
    def fit(self, n_epochs):
#'recall_score'
        self.learn.fit_one_cycle(n_epochs, self.lr_val, wd=self.hpd["wd"], 
                                 cbs=[EarlyStoppingCallback(monitor=self.hpd["cb_metric"], 
                                                            min_delta=0.01, 
                                                            patience=self.hpd["cb_patience"])
                                      ,SaveModelCallback(monitor=self.hpd["cb_metric"], min_delta=0.01)])

        self.learn.path = self.keep_path
        
        #self.roc_auc, self.valid_df, self.brier_score = get_metrics(self.learn)
    
        return self  

    def split_train_valid(self, valid_pct= 0.2):
        trn, vld  = RandomSplitter(seed = self.seed, valid_pct=valid_pct)(range_of(self.df))
        self.df_train = self.df.iloc[trn, :]
        self.df_valid = self.df.iloc[vld, :]         

    def resample_data(self):     
        self.split_train_valid()
        self.get_to(df=self.df_train, split_method=None)

        # retrieve xs and ys from tabularpandas object
        xs = self.to.items
        ys = self.to.ys

        resampling_dict = self.resampling_dict   

        t = self.target[0]
        ss = resampling_dict[t]["ss"]
        kn = resampling_dict[t]["kn"]

        print("resampling")
        x,y = resample_to(xs, ys, ss, kn, oversample_method =self.resampling)

        self.to.items = x
        self.to.ys = y   

        # new to object for validation with same metadata
        to_val = self.to.new(self.df_valid)
        to_val.process()
        self.to_valid = to_val

        #next, send to DLS train
        trn_dl = TabDataLoader(self.to, self.hpd ["batch_size"], shuffle=True, drop_last=True)
        val_dl = TabDataLoader(self.to_valid,self.hpd ["batch_size"])
        self.dls = DataLoaders(trn_dl, val_dl)
        

    def plot(self,roc_auc=None, valid_df =None, plots=['roc_auc','precision_recall']):                       
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(11,4), dpi =100,  facecolor='w', edgecolor='k')

        if roc_auc is None: roc_auc = self.roc_auc
        if valid_df is None: valid_df = self.valid_df

        if 'roc_auc' in plots:
            multi_label_roc_plot(roc_auc, valid_df, axs[0])

        if 'precision_recall' in plots:
            plot_precision_recall(valid_df, self.target, axs[1])
        fig.savefig('./mlflow_figs/metric_plot.png')    
        return self


    def get_test_preds(self, test_df, test_df_name, threshold_df=None,plots=['roc_auc', "precision_recall"]):

        """
        Test model with another dataframe. Requires similiar data structure

        """
        self.test_dataframes_dict[test_df_name]= test_df 

        to_test = self.learn.dls.train_ds.new(test_df)
        to_test.process()
        dl_test = TabDataLoader(to_test)
        self.dl_test = dl_test

        # to do: dissolve get_metrics -use as below (AP) on preds as funcs
        test_roc_auc, test_valid_df, test_brier_dict = get_metrics(self.learn, dl = dl_test)

        self.test_roc_auc =  test_roc_auc
        self.test_valid_df = test_valid_df
        self.test_brier_dict = test_brier_dict
        
        
        self.test_preds, self.test_ys = self.learn.get_preds(dl=dl_test)
        self.test_preds = self.test_preds[:,1]
        self.AP_score = average_precision_score(self.test_ys, self.test_preds)
        
       # look into what for
        self.val_preds, self.val_ys = self.learn.get_preds()
        self.val_preds = self.val_preds[:,1]
        
         #  for threshold-moving
        if threshold_df is not None:
            print("using threshold df")
            to_thres = self.learn.dls.train_ds.new(threshold_df)
            to_thres.process()
            dl_thres = TabDataLoader(to_thres)
            thres_preds, thres_ys = self.learn.get_preds(dl=dl_thres)
            self.test_prediction_threshold = sensivity_specifity_cutoff(thres_ys, thres_preds[:,1], method = "Fbeta-score", beta = self.hpd["f_beta"])
        else:
            self.test_prediction_threshold = sensivity_specifity_cutoff(self.val_ys, self.val_preds, method = "Fbeta-score", beta = self.hpd["f_beta"])
               
#        self.test_prediction_threshold = sensivity_specifity_cutoff(self.test_ys, self.test_preds)  

       #  get preds with threshold-moving for precision recall
        self.j_preds = (self.test_preds >= self.test_prediction_threshold)#.astype(bool)
        self.test_recall = recall_score(self.test_ys, self.j_preds)
        self.test_precision = precision_score(self.test_ys, self.j_preds)
        

        if self.test_results.empty:
            self.test_results = pd.DataFrame.from_dict(test_roc_auc, orient='index', columns =[test_df_name]).sort_index().transpose()
        else:
            self.test_results = pd.concat([self.test_results, pd.DataFrame.from_dict(test_roc_auc, orient='index', columns =[test_df_name]).sort_index().transpose()])

        if plots==['roc_auc', "precision_recall"]:
            self.plot(roc_auc = test_roc_auc, valid_df = test_valid_df)
        else:
            pass
            
    def get_metrics(learn, dl=None, plot_only = True):
        if dl:
            preds, y = learn.get_preds(dl=dl)
        else:
            preds, y = learn.get_preds()
            
        if learn.pile == "slp":
            preds = transform_preds(preds)
        else:
            pass
        
        y_names=learn.dls.y_names
        
        n_classes=len(y_names)

        valid = pd.DataFrame( )

        for x in range(n_classes):
            valid['y_true_'+(y_names[x])] = [i[x] for i in y.tolist()] 
            valid['y_score_'+(y_names[x])] = [i[x] for i in preds.tolist()] 
        
        roc_auc_dict={}
        brier_dict={}
        
        for i,j in zip(range(n_classes), y_names):
            l,AUC,h = roc_auc_ci((valid[('y_true_'+y_names[i])]), (valid[('y_score_'+y_names[i])]), positive=1)
            roc_auc_dict[j]=AUC
            b = brier_score_loss((valid[('y_true_'+y_names[i])]), (valid[('y_score_'+y_names[i])]))
            brier_dict[j]=b   
            
        return roc_auc_dict, valid, brier_dict

