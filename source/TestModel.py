import pandas as pd
from sklearn.preprocessing import StandardScaler
from CombinedClassifier import CombinedClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def TestModel(model, testData, showReport=True):
    (commodel, scaler, threshold, imputer) = model
    
    X_test = testData.drop(['positive','negative'],axis=1).fillna(0)
    y_test = testData['positive']
    
    X_test = X_test.set_index('PatientEncounterCSNID')
    X_test = pd.DataFrame(scaler.transform(X_test)    
                         , columns=X_test.columns, index=X_test.index)
    #X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
    y_prob=commodel.predict(X_test)
    y_pred=[1 if p>=threshold else 0 for p in y_prob]
    
    report = classification_report(y_test,y_pred)
    if showReport:
        print(report)
    
    fpr,tpr,thr = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test,y_pred,beta=2)
    precision = precision[1]
    recall = recall[1]
    beta=2
    fbeta = (1+beta*beta)*recall*precision/(beta*beta*precision+recall)
            
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec, prec)
    
    results={}
    results['roc']={}
    results['roc']['fpr']=fpr
    results['roc']['tpr']=tpr
    results['roc']['thr']=thr
    results['roc']['auc']=roc_auc
    results['prcurve']={}
    results['prcurve']['precisions']=prec
    results['prcurve']['recalls']=rec
    results['prcurve']['auc']=pr_auc
    results['prcurve']['precision']=precision
    results['prcurve']['recall']=recall
    results['prcurve']['f2score']=fbeta
    results['y_test']=y_test
    results['y_prob']=y_prob
    results['y_pred']=y_pred
    
    return results