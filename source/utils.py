import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# Import train_test_split function

### Evaluate model
from sklearn.metrics import \
        roc_curve, precision_recall_curve, auc, \
        make_scorer, recall_score, accuracy_score, precision_score, average_precision_score, \
        confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


def construct_results(y_true, y_score, dateDF, X_test, y_score_thresholded=None, write_to_file=None):
    """ Constructs results dataframe """
    """Needed structure: [OrderDate, PatientEncounterCSNID, PatientID, label, score, thresholded_score]"""
    

    dateDF2 = dateDF.loc[dateDF.index.isin(X_test.index)]
    X_test_res = X_test.copy()
    X_test_res['label']=y_true.values
    X_test_res['score']=y_score
    if y_score_thresholded is not None:
            X_test_res['thresholded_score']=y_score_thresholded
    result = X_test_res[['label','score']].merge(dateDF2, left_index=True, right_index=True)\
    .rename(columns={'MinEffectiveDate' : 'OrderDate'})[['PatientID', 'OrderDate', 'label', 'score']]
    
    if write_to_file is not None:
        result.to_csv(write_to_file+'_output.csv')
    
    return result


def plot_confusion_matrix(y_pred, y_true, ax=None, display_labels=None):
    if ax is None:
        f, ax = plt.subplots()
       
    confmat = confusion_matrix(
        y_pred=y_pred, y_true=y_true,
        normalize="true")
    
    if display_labels is None:
        display_labels = ["EDamb", "EDnoamb", "Admission"]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confmat,
        display_labels=display_labels)
    disp.plot(include_values=True,
              cmap=plt.cm.Blues,
              ax=ax,
              xticks_rotation="vertical",
              values_format=".2g")

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)

    disp.ax_.set_title(f"Confusion Matrix | Acc: {acc:.2f}")
    return disp.ax_

def plot_precision_recall(y_true, y_pred, ax=None, c=""):
    if ax is None:
        f, ax = plt.subplots()
    p, r, thresholds = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true=y_true, y_score=y_pred)

    ax.step(r, p, alpha=0.4,
            where='post', label=f"{c} | {ap_score:.2f}")  #{labels.sum()}")
    return ax

def plot_roc_auc(y_true, y_pred, ax=None, c=""):
    if ax is None:
        f, ax = plt.subplots()
    auc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr, alpha=0.5, label=f"{c} | AUC {auc_score:.2f}")
    return ax

def plot_results(results, title=None, n_class=3):
    """ Assumes results have keys Target, EDamb_prob, EDnoamb_prob, Admission_prob """
    if n_class == 3:
        class_map = {
            "EDamb_prob": 1,
            "EDnoamb_prob": 2,
            "Admission_prob": 3
        }

    else:
        class_map = {
            "None_prob": 0,
            "EDamb_prob": 1,
            "EDnoamb_prob": 2,
            "Admission_prob": 3
        }

    disp_labels = [x.split("_")[0] for x in class_map]

        
    f, axarr = plt.subplots(1, 3, figsize=(24, 6))
    for c, cval in class_map.items():
        labels = (results["Target"] == cval).astype(int)
        # AUC
        axarr[0] = plot_roc_auc(labels, results[c], ax=axarr[0], c=c)
        
        
        # PR
        axarr[1] = plot_precision_recall(labels, results[c], ax=axarr[1], c=c)
    #     plt.fill_between(r, p, step='post', alpha=0.2,
    #                      color='b')

    # Confusion matrix
    axarr[2] = plot_confusion_matrix(y_pred=results["Predicted"], y_true=results["Target"], ax=axarr[2], display_labels=disp_labels)

    axarr[0].axis([0,1.01 , 0, 1.01])
    axarr[0].set_xlabel('False Positive Rate')
    axarr[0].set_ylabel('True Positive Rate')
    axarr[0].set_title('ROC')
    axarr[1].set_ylim([0.0, 1.01]);
    axarr[1].set_xlim([0.0, 1.01]);
    axarr[1].set_xlabel('Recall');
    axarr[1].set_ylabel('Precision');
    axarr[1].set_title('Precision Recall');


    axarr[0].legend()
    axarr[1].legend()

    if title:
        f.suptitle(f"{title} | {len(results)} total patients")

    return axarr


def plot_summary_results(results_dict, title=None):

    class_map = {
        "EDamb_prob": 1,
        "EDnoamb_prob": 2,
        "Admission_prob": 3
    }

    f, axarr = plt.subplots(2, 3, figsize=(24, 12))
    for i, (c, cval) in enumerate(class_map.items()):
        for feature_space, results in results_dict.items():
            labels = (results["Target"] == cval).astype(int)
            # AUC
            auc_score = roc_auc_score(labels, results[c])
            fpr, tpr, thresholds = roc_curve(labels, results[c])
            axarr[0, i].plot([0, 1], [0, 1], 'k--')
            axarr[0, i].plot(fpr, tpr, linewidth = 1, alpha=0.5,
                          label=f"{feature_space} | AUC {auc_score:.2f}")

            # PR
            p, r, thresholds = precision_recall_curve(labels, results[c])
            ap_score = average_precision_score(labels, results[c])
            axarr[1, i].step(r, p, alpha=0.4,
                     where='post', label=f"{feature_space} | {ap_score:.2f}")
        #     plt.fill_between(r, p, step='post', alpha=0.2,
        #                      color='b')

        axarr[0, i].axis([0,1.01 , 0, 1.01])
        axarr[0, i].set_xlabel('False Positive Rate')
        axarr[0, i].set_ylabel('True Positive Rate')
        axarr[0, i].set_title(f'{c} ROC')
        axarr[1, i].set_ylim([0.0, 1.01]);
        axarr[1, i].set_xlim([0.0, 1.01]);
        axarr[1, i].set_xlabel('Recall');
        axarr[1, i].set_ylabel('Precision');
        axarr[1, i].set_title(f'{c} Precision Recall | {labels.sum()} samples');


        axarr[0, i].legend(loc="lower right")
        axarr[1, i].legend(loc="lower right")

    if title:
        f.suptitle(f"{title} | {len(results)} total patients")

    return axarr

def filter_results(results, idx):
    return {k: v[idx] for k, v in results.items()}
