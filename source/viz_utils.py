from sklearn.metrics import \
    roc_curve, precision_recall_curve, \
    roc_auc_score, average_precision_score, auc
from sklearn.metrics import confusion_matrix, brier_score_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bootstrap(y_true, y_pred, n=500):
    res = np.vstack([y_true, y_pred]).T

    fprs, tprs = [], []
    precisions, recalls = [], []

    fpr_actual, tpr_actual, _ = \
        roc_curve(res[:, 0], res[:, 1])

    precision_actual, recall_actual, _ = \
        precision_recall_curve(y_true, y_pred)

    for i in range(n):
        idx = np.random.choice(len(res), size=len(res), replace=True)
        fpr, tpr, _ = roc_curve(res[idx, 0], res[idx, 1])
        fprs.append(fpr)
        tprs.append(tpr)
        precision, recall, _ = precision_recall_curve(res[idx, 0], res[idx, 1])
        precisions.append(precision)
        recalls.append(recall)
        
    return {
        "actual": (fpr_actual, tpr_actual), "sim": (fprs, tprs)
    }, {
        "actual": (recall_actual, precision_actual), "sim": (recalls, precisions)
    }

def ci_bootstrap(y_true, y_pred, threshold=0.5, n=500):
    res = np.vstack([y_true, y_pred]).T
    fps, tps, tns, fns = [], [], [], []
    tn, fp, fn, tp = \
        confusion_matrix(y_true, [1 if y>=threshold else 0 for y in y_pred]).ravel()    
    #briers = []
    #brier = brier_score_loss(y_true=y_true, y_prob=y_pred)
    #briers.append(brier)
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)
    tps.append(tp)
    for i in range(n):
        idx = np.random.choice(len(res), size=len(res), replace=True)
        tn, fp, fn, tp = \
            confusion_matrix(res[idx, 0], res[idx, 1] > threshold).ravel()
        #brier = brier_score_loss(y_true=res[idx, 0], y_prob=res[idx, 1])
        #briers.append(brier)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        
    return pd.DataFrame({"FP": fps, "TP": tps, "FN": fns, "TN": tns})#, "Brier": briers})

def mpl_plot(roc_dict, pr_dict, saveFigs=False, figName='', legend=True):
    aurocs, auprcs = [], []
    f, axarr = plt.subplots(1, 2, figsize=(20, 8))
    for fpr, tpr in zip(*roc_dict["sim"]):
        axarr[0].plot(fpr, tpr, 'b', alpha=0.05)
        aurocs.append(auc(fpr, tpr))
        axarr[0].set_ylim((-0.1,1.1))
        axarr[0].grid()
    for recall, precision in zip(*pr_dict["sim"]):
        axarr[1].step(recall, precision, 'b', where='post', alpha=0.05)
        auprcs.append(auc(recall, precision))
        axarr[1].set_ylim((-0.1,1.1))
        axarr[1].grid()
        
    auroc_bounds = (np.percentile(np.array(aurocs), 2.5), np.percentile(np.array(aurocs), 97.5))
    auprc_bounds = (np.percentile(np.array(auprcs), 2.5), np.percentile(np.array(auprcs), 97.5))
    roc_auc = auc(*roc_dict["actual"])
    pr_auc = auc(*pr_dict["actual"])
    axarr[0].plot(*roc_dict["actual"], 'red', label=f"AUROC: {roc_auc:.2f} ({auroc_bounds[0]:.2f}, {auroc_bounds[1]:.2f})")
    axarr[1].step(*pr_dict["actual"], 'red', where='post', label=f"AUPRC: {pr_auc:.2f} ({auprc_bounds[0]:.2f}, {auprc_bounds[1]:.2f})")
    axarr[0].set_xlabel("1-Specificity\n(False Positive Rate)")
    axarr[0].set_ylabel("Sensitivity \n(True Positive Rate)")
    axarr[1].set_xlabel("Recall (Sensitivity)")
    axarr[1].set_ylabel("Precision (PPV)")
    if saveFigs:
        print('saving nl fig')
        f.savefig(figName+'_noLegend.png')
    if legend:
        axarr[0].legend()
        axarr[1].legend()
    if saveFigs:
        print('saving l fig')
        f.savefig(figName+'_Legend.png')
#     plt.tight_layout()
    return f, axarr


def plotly_plot(roc_dict, pr_dict):
    fig = plotly.subplots.make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=roc_dict["actual"][0], 
                   y=roc_dict["actual"][1],
                   name="ROC",
                   line_color='red'),
        row=1, col=1
    )
    for fpr, tpr in zip(*roc_dict["sim"]):
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, 
                       fillcolor='gray', 
                       opacity=0.05,
                       showlegend=False,
                       hoverinfo='skip',
                       line_color='blue'),
            row=1, col=1
        )
        
    fig.add_trace(
        go.Scatter(x=pr_dict["actual"][0], 
                   y=pr_dict["actual"][1],
                   name="ROC",
                   line_color='red'),
        row=1, col=2
    )
    for rec, pr in zip(*pr_dict["sim"]):
        fig.add_trace(
            go.Scatter(x=rec, y=pr, 
                       fillcolor='gray', 
                       opacity=0.05,
                       showlegend=False,
                       hoverinfo='skip',
                       line_color='blue'),
            row=1, col=2
        )
        
    return fig

def plot_results(y_true, y_pred, saveFigs=False, figName='', legend=True):
    roc_dict, pr_dict = bootstrap(y_true, y_pred, n=100)
    mpl_plot(roc_dict, pr_dict, saveFigs=saveFigs, figName=figName, legend=legend)

    """plt.figure(figsize=(10, 5))
    plot_data = [y_pred[list(y_true.values) == 0], y_pred[list(y_true.values) == 1]]
    plt.violinplot(plot_data, #widths=[0.5, y_true.sum() / len(y_true) * 0.5], 
                   showmedians=True, showextrema=True) #, alpha=0.6) quantiles=[25, 75], 
    plt.xticks([1, 2], ["Negative", "Positive"])
    plt.ylim([0, 1.1])
    plt.ylabel("Probability")
#     plt.hist(y_pred[y_true == 0], alpha=0.6, label="negative")
#     plt.hist(y_pred[y_true == 1], alpha=0.6, label="positive")
#     plt.legend()
    plt.tight_layout()"""

def plot_metrics_df(df):
    f, ax = plt.subplots(figsize=(10, 5))
    ax = df.boxplot(
                column=["sensitivity", "specificity", "ppv", "npv", "fpr"], ax=ax) # , "Brier", 
    return f, ax

def plot_cis(y_true, y_pred, threshold_to_use=0.5, plot=True, ax=None):
    cm_df = ci_bootstrap(y_true, y_pred, threshold=threshold_to_use)
    metrics_df = cm_df.copy()
    metrics_df["sensitivity"] = cm_df["TP"] / (cm_df["TP"] + cm_df["FN"])
    metrics_df["specificity"] = cm_df["TN"] / (cm_df["TN"] + cm_df["FP"])
    metrics_df["ppv"] = cm_df["TP"] / (cm_df["TP"] + cm_df["FP"])
    metrics_df["npv"] = cm_df["TN"] / (cm_df["TN"] + cm_df["FN"])
    metrics_df["fpr"] = cm_df["FP"] / (cm_df["FP"] + cm_df["TN"])
    metrics_df["tpr"] = metrics_df["sensitivity"]
    for met in ['sensitivity', 'specificity', 'ppv', 'npv', 'fpr', 'tpr']:
        met_med=metrics_df[met].median()
        met_low=metrics_df[met].quantile(0.025)
        met_high=metrics_df[met].quantile(0.975)
        print('%s : %.2f 95%%CI (%.2f, %.2f)' % (met, met_med, met_low, met_high))
    if plot:
        f, ax = plot_metrics_df(metrics_df)
        ax.set_title(f"Selected Threshold: {threshold_to_use:.2f}")
        ax.set_ylim((-0.1,1.1))
        return metrics_df, ax
    return metrics_df

from sklearn.inspection import permutation_importance


def plot_feature_importance(model, testData, figPath=None, n_repeats=10):
    X_test=testData.drop(['positive','negative'],axis=1).fillna(0).set_index('PatientEncounterCSNID')
    y_test = testData['positive']
    X_test = pd.DataFrame(model[1].transform(X_test), columns=X_test.columns, index=X_test.index)
    sns.set_style('whitegrid', {'axes.apines.bottom': True,
                               'axes.apines.left': True,
                               'axes.apines.right': False,
                               'axes.apines.top': False,
                               'axes.edgecolor': '.0'})
    sns.set_context('talk')
    sns.set_palette('colorblind')

    result = permutation_importance(model[0], X_test, y_test, n_repeats=n_repeats,
                                    random_state=42, n_jobs=2)
    #sorted_idx = result.importances_mean.argsort()
    """.rename(columns={\
        'PLATELET COUNT, AUTO':'Platelet Count',\
        'D-DIMER':'D-Dimer'})"""
    #idx_grouped=[10, 9, 8, 7, 14, 11, 6, 5, 4, 3, 2, 1, 0, 13, 12]
    idx_grouped=[4, 5, 9, 11, 10, 8, 7, 6, 3, 2, 1, 0, 13, 12]
    print(X_test.columns)
    #print(X_test.columns[[x for x in range(len(result.importances))]])
    print(X_test.columns[idx_grouped])
    fig, ax = plt.subplots(figsize=(10,10))
    ax.boxplot(result.importances[idx_grouped].T,
               vert=False, labels=X_test.columns[idx_grouped])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()
    if figPath is not None:
        plt.savefig(figPath)