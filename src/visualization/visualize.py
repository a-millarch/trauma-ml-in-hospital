import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict

from sklearn.metrics import roc_curve, auc, roc_auc_score, brier_score_loss, precision_recall_curve, f1_score, precision_score, recall_score
from math import sqrt

def plot_box_kde(df, dep, y):
    
    fig,axs = plt.subplots(1,2, figsize=(11,5))
    sns.boxenplot(data=df, x=dep, y=y, ax=axs[0])
    sns.kdeplot(data=df[df[dep]==0], x=y, ax=axs[1], label="0");
    sns.kdeplot(data=df[df[dep]==1], x=y, ax=axs[1], label="1");
    fig.legend()
    
    mask_0 = (df[dep] == 0)
    mask_1 = (df[dep] == 1)
    
    print("is null in total\t\t", "%.2f" % float(df[y].isna().sum()/len(df)*100), "%" )
    print("is null in",dep,"== 0\t", "%.2f"% float(df[mask_0][y].isna().sum()/len(df[mask_0])*100), "%" )
    print("is null in",dep,"== 1\t", "%.2f" % float(df[mask_1][y].isna().sum()/len(df[mask_1])*100), "%" )
    

def to_tiff(plt, title, tr):
    from PIL import Image
    import io
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png")

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(tr+'/reports/export/'+title+".tiff")
    plt.savefig(tr+'/reports/export/'+title+".png", format="png")
    png1.close()
    

def plot_fti(fti, threshold=0.0005, save_file=False ):
    fti= fti[fti['imp']>threshold].sort_values('imp').plot('cols', 'imp', 'barh', figsize=(10,8));
    if save_file:
        fti_plot = fti.get_figure()
        fti_plot.savefig('./mlflow_figs/fti_plot.png')    
    
    
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



 def plot_ranfor_roc(val_y, predictions, ax):
    fpr, tpr, _ = roc_curve(val_y, predictions)#[:,1])
    auc = roc_auc_score(val_y, predictions)
   # print(auc)
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
        return fti_plot

  