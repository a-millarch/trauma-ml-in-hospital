from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss, precision_recall_curve, f1_score, precision_score, recall_score
from math import sqrt
from collections import OrderedDict

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline

import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import numpy as np
import torch

import matplotlib.pyplot as plt

def transform_preds (preds):
        indices=torch.tensor([1])
        preds = torch.index_select(preds, 1, indices)
        return preds


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)#, multi_class='ovr')
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, AUC, upper)


def dk_test_roc_auc_ci(AUC, dep_var="DECEASED"):
    # no. pos cases and negative static due to static test data. 
    #change n1 to N posive and n2 to N negative if changing test dataset.
    if dep_var = "LOS48HOURSPLUS":
        N1 = 215 
        N2 = 92
    elif dep_var = "DECEASED":
        N1= 22
        N2= 285

    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return [lower, AUC, upper]


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



def multi_label_roc_plot(roc_auc_dict, valid, ax):
    
    roc_values = list(roc_auc_dict.values())
    y_names = list(roc_auc_dict.keys())
    
    d=dict(list(enumerate(roc_values)))
    dd = OrderedDict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    
    roc_ordered=dd.keys()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
 
    roc_values=[]

    for i in range(len(y_names)):
        fpr[i], tpr[i], _ = roc_curve((valid[('y_true_'+y_names[i])]), (valid[('y_score_'+y_names[i])]))
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_values.append(roc_auc[i])

    

    for i in(roc_ordered):
        ax.plot(fpr[i], tpr[i], lw=1,
        label=y_names[i]+' (area = {1:0.3f})'''.format(i, roc_auc[i]))
    ax.grid(visible=1, alpha=0.3)
    ax.legend(loc="lower right", prop={'size':9})
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("ROC")



def plot_precision_recall(valid_df, target, ax):
        #plt.figure(num=None, figsize=(7, 6), dpi=100, facecolor='w', edgecolor='k')
        if isinstance(target, str):
            target = [target,]
            
        for i in target:
            y_true = valid_df["y_true_"+i]
            y_scores = valid_df["y_score_"+i]

            precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
            #

            ax.plot(recall, precision, label = i, lw=1)

        ax.grid(visible=1, alpha=0.3)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.legend(loc="upper right",  prop={'size':9})
        ax.set_title("Precision Recall")

      
        
def resample_to(xs,ys , ss, kn, oversample_method='SMOTE', undersample_method=None, return_original_df=True):
    """ Applies synthetic minority oversampling technique (SMOTE) to xs and ys from tabularpandas object"""
    # instantiate and initialize sklearn function
    if oversample_method == 'SMOTE':
        print("smote")
        resample =SMOTE(sampling_strategy =ss ,k_neighbors=kn)
        
    elif oversample_method == 'BorderlineSMOTE':
        resample =BorderlineSMOTE(sampling_strategy =ss ,k_neighbors=kn)    
        
    elif oversample_method == 'SVMSMOTE':
        #resample =SVMSMOTE(sampling_strategy =ss ,k_neighbors=kn) 
        resample =SVMSMOTE() 
        
    elif oversample_method == 'ADASYN':
        resample =ADASYN(sampling_strategy =ss )    
    
    elif oversample_method == 'SMOTETomek':
        #o = SMOTE(sampling_strategy =ss ,k_neighbors=kn)
        #u = TomekLinks()
        #resample =SMOTETomek(sampling_strategy =ss , smote=o, tomek=u)
        resample = SMOTETomek(sampling_strategy=ss, n_jobs=-1)
        
    elif oversample_method == 'SMOTENearMiss':
        
        class_0 = int(len(xs)*0.7)
        class_1 = int(ys.sum())
        resample = make_pipeline(
            #NearMiss(sampling_strategy=0.3, version=1 ),
            #NearMiss(sampling_strategy={1:class_1, 0:class_0}, version=1 )
            SMOTE(sampling_strategy =ss, k_neighbors=kn),
            NearMiss(sampling_strategy={0:class_0}, version=1 )
            #NearMiss(sampling_strategy=0.5, version=1 )
        )

    else:
        print("invalid oversample_method provided")
    # assign resampled objects
    x,y = resample.fit_resample(xs, ys)
    #return new rows to TabularPandas object
    if return_original_df is False:
        x = x[len(xs):]
        y = y[len(ys):]
    return x,y

def get_class_weights(df, target):
    """
    Returns class weight from dataframe with binary dependent variable(target)

                wj=n_samples / (n_classes * n_samplesj)      
    where:
        wj is the weight for each class(j signifies the class)
        n_samplesis the total number of samples or rows in the dataset
        n_classesis the total number of unique classes in the target
        n_samplesjis the total number of rows of the respective class
    """
    class_count_df = df.groupby(target).count() 
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0] 
    w_0 = (n_0 + n_1) / (2.0 * n_0)
    w_1 = (n_0 + n_1) / (2.0 * n_1) 
    class_weights=torch.FloatTensor([w_0, w_1])

    print("[0,1] weighted as ",class_weights)
    return class_weights    
    
def set_seeds(seed):	#seed = 42
    if seed is None:
        seed = np.random.randint(1000)
    # python RNG
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)
def sensivity_specifity_cutoff(y_true, y_score, method="Youden", beta=None):
   
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    if method == "Youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx]
        print('Best Threshold: {}'.format(threshold))
        
    elif method == "F-score":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        df_recall_precision = pd.DataFrame({'Precision':precision[:-1],
                                    'Recall':recall[:-1],
                                    'Threshold':thresholds})
        fscore = (2 * precision * recall) / (precision + recall)  
        # Find the optimal threshold
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits = 4)
        fscoreOpt = round(fscore[index], ndigits = 4)
        recallOpt = round(recall[index], ndigits = 4)
        precisionOpt = round(precision[index], ndigits = 4)
        print('Best Threshold: {}'.format(thresholdOpt))
        
        threshold = thresholdOpt
        
    elif method == "Fbeta-score":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        df_recall_precision = pd.DataFrame({'Precision':precision[:-1],
                                    'Recall':recall[:-1],
                                    'Threshold':thresholds})
        if beta is None:
            beta =2
        fscore = ((1+pow(beta,2)) * precision*recall)/(pow(beta,2)*precision+recall)

        # Find the optimal threshold
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits = 4)
        fscoreOpt = round(fscore[index], ndigits = 4)
        recallOpt = round(recall[index], ndigits = 4)
        precisionOpt = round(precision[index], ndigits = 4)
        print('Best Threshold: {}'.format(thresholdOpt))
        #print('Recall: {}, Precision: {}'.format(recallOpt, precisionOpt))
        
        threshold = thresholdOpt
    
    return threshold

def precision_recall_from_j_stat(model, val_xs, val_ys, threshold=None):
    predictions = model.predict_proba(val_xs)#[:,1]
    pred_pos = predictions[:,1]
    
    if threshold is None:
        threshold = sensivity_specifity_cutoff(val_ys, pred_pos)
        
    j_preds = (model.predict_proba(val_xs)[:,1] >= threshold).astype(bool)
    recall = recall_score(np.array(val_ys), j_preds)
    precision = precision_score(np.array(val_ys), j_preds)
    return precision, recall

def F1(precision, recall):
    F1 = 2*((precision * recall) / (precision + recall))
    return F1

def Fbeta(precision, recall, beta):
    Fbeta = (1+pow(beta,2))*((precision*recall)/((pow(beta, 2)*precision)+recall))
    return Fbeta