from sklearn.metrics import mean_absolute_error
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, recall_score,auc, average_precision_score
import pandas as pd
from matplotlib import pyplot as plt


def proc_data(df, cats, conts):
    for cat in cats:
        df[cat] = df[cat].fillna(df[cat].mode())
        df[cat] = pd.Categorical(df[cat])
    for cont in conts:
        df[cont] = df[cont].fillna(float(df[cont].mode()[0]))   
        #except: df[cont] = df[cont].fillna(df[cont].mode())
    return df
        
def xs_y(df, cats, conts, dep):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None       

def create_trn_val(df, cats, conts, dep):
    df = proc_data(df, cats, conts)
    trn_df,val_df = train_test_split(df[[dep,]+cats+conts], test_size=0.2)
    trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
    val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

    trn_xs,trn_y = xs_y(trn_df, cats, conts, dep)
    val_xs,val_y = xs_y(val_df, cats, conts, dep)
    
    return trn_xs, trn_y, val_xs, val_y 
   
def create_ranfor(df, cats, conts, dep):
    trn_xs, trn_y, val_xs, val_y = create_trn_val(df, cats, conts, dep)

    rf = RandomForestClassifier(100, min_samples_leaf=5)
    rf.fit(trn_xs, trn_y);
    print("mean abs err:",mean_absolute_error(val_y, rf.predict(val_xs)))

    predictions = rf.predict_proba(val_xs)[:,1]
    roc_auc = roc_auc_score(val_y, predictions)
    
    fti = pd.DataFrame(dict(cols=trn_xs.columns, imp=rf.feature_importances_))
    multiplot(val_y, predictions)
    
    rf.val_xs = val_xs
    rf.val_ys = val_y
    rf.trn_xs =trn_xs
    rf.trn_ys = trn_y
    return rf, predictions, roc_auc, fti

def multiplot(ys, preds):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(11,4), dpi =100 ,  facecolor='w', edgecolor='k')
    plot_ranfor_roc(ys, preds, axs[0])
    rf_plot_precision_recall(ys, preds, axs[1])
    

def rf_test_preds(rf,  test_xs,test_ys, plot=False):  
    test_preds = rf.predict_proba(test_xs)
    test_preds_pos = test_preds[:,1]
    
    if plot: multiplot(test_ys, test_preds_pos)
    else:pass
    return test_preds_pos, test_ys

def proc_split(test_df, cats, conts, dep):
    proc_data(test_df, cats, conts)
    test_df[cats] = test_df[cats].apply(lambda x: x.cat.codes)
    
    test_xs,test_ys = xs_y(test_df, cats, conts, dep)
    return test_xs,test_ys 
 
 
 
def plot_ranfor_roc(val_y, predictions, ax):
    fpr, tpr, _ = roc_curve(val_y, predictions)#[:,1])
    #plt.clf()
    auc = roc_auc_score(val_y, predictions)
   # print(auc)
    #ax.plot(fpr, tpr, label = '(area = {:.3f})'''.format(auc))
    ax.plot(fpr, tpr, label = "AUC: "+ str(auc)[:5])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC curve')
    ax.grid(visible=1, alpha=0.3)
    ax.legend(loc="lower right", prop={'size':9})
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
def rf_plot_precision_recall(ys, preds, ax):
    precision, recall, thresholds = precision_recall_curve(ys, preds, pos_label=1)
    #
    #f1 = f1_score(ys, preds)
    ax.plot(recall, precision, lw=1, label="f1")

    ax.grid(visible=1, alpha=0.3)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend(loc="upper right",  prop={'size':9})
    ax.set_title("Precision Recall")

    
def plot_fti(fti, threshold=0.0005, save_file=False ):
    fti= fti[fti['imp']>threshold].sort_values('imp').plot('cols', 'imp', 'barh', figsize=(10,8));
    
    if save_file:
        fti_plot = fti.get_figure()
        #fti_plot.savefig('./mlflow_figs/fti_plot.png')    
        return fti_plot
   # return fti
   
  