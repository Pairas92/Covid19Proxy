from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.preprocessing import StandardScaler
from CombinedClassifier import CombinedClassifier
from sklearn.metrics import roc_auc_score
import tqdm
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer


def TrainModel(trainData, testData, reIncludeVal=True, tuneThreshold=True, iterations=5, plotRocFigs=False, plotSubmodelFigs=False, plotpath=None, testRF=False, testLR=False, minThr=15, maxThr=31, random_state=1, beta=2):
    randSt=np.random.RandomState(random_state)
    expnum='OnlyCBCandDem'
    experiments={}
    expnames=['full', 'OnlyCBCandDem', 'NoDDIMER', 'NoFerritin', 'NoLDH', 'NoCRP']
    colToDrop=[]
    
    submodelplots={}
    classifiers=['comb', 'rf', 'logreg']
    classifier = 'comb'

    if plotRocFigs:
        f, axarr = plt.subplots(1, 2, figsize=(20, 5))
        axarr[0].set_title('Receiver Operating Characteristics', fontsize=18)
        axarr[0].set_ylim((-0.1,1.1))
        axarr[1].set_title('Precision-Recall Curve', fontsize=18)
        axarr[1].set_ylim((-0.1,1.1))

        
    X_train = trainData.drop(['positive','negative']+colToDrop,axis=1).fillna(0).replace([np.inf, -np.inf], 0)
    y_train = trainData['positive']
    X_test = testData.drop(['positive','negative']+colToDrop,axis=1).fillna(0).replace([np.inf, -np.inf], 0)
    y_test = testData['positive']
        
    scaler = StandardScaler()
    scaler.fit(X_train.drop(['PatientEncounterCSNID'], axis=1))
    X_train = X_train.set_index('PatientEncounterCSNID')
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    imputer = KNNImputer(n_neighbors=2)
    #imputer = imputer.fit(X_train)
    #rfr = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, n_jobs=-1)
    #imputer = IterativeImputer(random_state=0, estimator=rfr, verbose=1).fit(X_train)
    #X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)

    X_test = X_test.set_index('PatientEncounterCSNID')
    X_test = pd.DataFrame(scaler.transform(X_test)    
                         , columns=X_test.columns, index=X_test.index)  # apply same transformation to test data
    #X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    
        
    print("Optimizing decision threshold...")
    thresholds = [x/100 for x in range(minThr,maxThr)]
    bestThreshold = thresholds[0]
    bestfscore = 0
    #for threshold in tqdm.tqdm(thresholds):
    for threshold in thresholds:
        precisions = []
        recalls = []
        fbetas = []
        aucs = []
        for i in range(iterations):
            comclf = CombinedClassifier(randomState=randSt.randint(1,100))
            commodel = comclf.fit(X_train,y_train)
            y_prob=commodel.predict(X_test)
            y_pred=[1 if p>=threshold else 0 for p in y_prob]
            """#Random Forest#
                if classifier=='rf':
                    rfclf = RandomForestClassifier(random_state = 1,
                                                      n_estimators = 100,
                                                      max_depth = None, 
                                                      min_samples_split = 2,  min_samples_leaf = 1)
                    rfclf.fit(X_train, y_train)
                    y_prob=rfclf.predict_proba(X_test)[:,1]
                    y_pred=rfclf.predict(X_test)
                #Logistic Regression#
                if classifier=='logreg':
                    logmodel = LogisticRegression(solver='liblinear')
                    logmodel.fit(X_train,y_train)
                    y_prob = logmodel.decision_function(X_test)
                    y_pred = logmodel.predict(X_test)
                #print(classification_report(y_test,y_pred))
                #print(fbeta_score(y_test,y_pred,beta=2,average='binary'))"""

            fpr,tpr,thr = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            precision, recall, fscore, support = precision_recall_fscore_support(y_test,y_pred,beta=3)
            precisions.append(precision)
            recall=recall[1]
            recalls.append(recall)
            fbeta = (1+beta**2)*recall*precision/(beta**2*precision+recall)
            fbetas.append(fbeta)

                
        m_precision = np.mean(precisions)
        m_recall = np.mean(recalls)
        m_fbeta = np.mean(fbetas)
        #print('%.2f : P %.2f R %.2f F %.2f' % (threshold,m_precision,m_recall,m_fbeta))
        if m_fbeta>bestfscore:
            bestfscore=m_fbeta
            bestThreshold=threshold
        print('Threshold: %.2f     Precision: %.2f     Recall: %.2f     Fbeta: %.2f' % (threshold, m_precision, m_recall, m_fbeta))

        
    if reIncludeVal:    
        X_train = pd.concat([trainData, testData]).drop(['positive','negative']+colToDrop,axis=1).fillna(0).replace([np.inf, -np.inf], 0)
        y_train = pd.concat([trainData, testData])['positive']

        scaler = StandardScaler()
        scaler.fit(X_train.drop(['PatientEncounterCSNID'], axis=1))
        X_train = X_train.set_index('PatientEncounterCSNID')
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)  
        comclf = CombinedClassifier(randomState=randSt.randint(1,100))
        commodel = comclf.fit(X_train,y_train)
    
    return (commodel, scaler, bestThreshold, imputer)