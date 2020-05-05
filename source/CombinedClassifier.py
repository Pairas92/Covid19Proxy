from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, auc
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class CombinedClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, randomState=5, threshold=0.5):
        """
        Called when initializing the classifier
        """
        self.threshold=threshold
        self.classifiers=['svm', 'rf', 'log', 'xgb', 'sgd', 'mlp', 'ada']
        self.randomState=randomState
        self.randSt=np.random.RandomState(self.randomState)

    #def set_threshold(self, threshold):
    #    self.threshold=threshold

    def fit(self, X_train, y_train=None):
        self.clfs = {}
        self.clfs['logmodel'] = LogisticRegression()
        self.clfs['logmodel'].fit(X_train,y_train)
        #self.clfs['linmodel'] = LinearRegression()
        #self.clfs['linmodel'].fit(X_train,y_train)       
        self.clfs['svmmodel'] = svm.SVC(probability=True)
        self.clfs['svmmodel'].fit(X_train,y_train)       
        self.clfs['rfmodel'] = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt', random_state=self.randSt.randint(1,100))
        self.clfs['rfmodel'].fit(X_train,y_train)       
        self.clfs['mlpmodel'] = mlpmodel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),random_state=self.randSt.randint(1,100))
        self.clfs['mlpmodel'].fit(X_train,y_train)       
        self.clfs['sgdmodel'] = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=500,random_state=self.randSt.randint(0,100))
        self.clfs['sgdmodel'].fit(X_train,y_train)       
        self.clfs['xgbmodel'] = XGBClassifier(random_state=self.randSt.randint(0,100))
        self.clfs['xgbmodel'].fit(X_train,y_train)       
        self.clfs['adamodel'] = adamodel = AdaBoostClassifier(n_estimators=100,random_state=self.randSt.randint(1,100))
        self.clfs['adamodel'].fit(X_train,y_train)       
        
        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( True if x >= self.threshold else False )

    def predict(self, X_test, y_test=None, threshold=None):
        if threshold is not None:
            self.threshold=threshold
        self.pred={}
        self.predneg={}
        log_probs = self.clfs['logmodel'].predict_proba(X_test)[:,1]
        log_probs=((log_probs-np.mean(log_probs))/np.std(log_probs))
        log_probs=(log_probs+max(abs(log_probs)))
        log_probs=log_probs/max(abs(log_probs))
        self.pred['log'] = log_probs
        
        #self.pred['lin'] = self.clfs['linmodel'].predict(X_test)
        svm_probs = self.clfs['svmmodel'].predict_proba(X_test)[:,1]
        svm_probs=((svm_probs-np.mean(svm_probs))/np.std(svm_probs))
        svm_probs=(svm_probs+max(abs(svm_probs)))
        svm_probs=svm_probs/max(abs(svm_probs))
        self.pred['svm'] = svm_probs
        self.pred['rf'] = self.clfs['rfmodel'].predict_proba(X_test)[:,1]
        self.pred['mlp'] = self.clfs['mlpmodel'].predict_proba(X_test)[:,1]

        sgd_probs = self.clfs['sgdmodel'].predict_proba(X_test)[:,1]
        sgd_probs=((sgd_probs-np.mean(sgd_probs))/np.std(sgd_probs))
        sgd_probs=(sgd_probs+max(abs(sgd_probs)))
        sgd_probs=sgd_probs/max(abs(sgd_probs))
        self.pred['sgd'] = sgd_probs
        self.pred['xgb'] = self.clfs['xgbmodel'].predict_proba(X_test)[:,1]
        self.pred['ada'] = self.clfs['adamodel'].predict_proba(X_test)[:,1]
        self.predneg['log'] = self.clfs['logmodel'].predict_proba(X_test)[:,0]
        #self.predneg['lin'] = [1-y for y in self.clfs['linmodel'].predict(X_test)]
        self.predneg['svm'] = self.clfs['svmmodel'].predict_proba(X_test)[:,0]
        self.predneg['rf'] = self.clfs['rfmodel'].predict_proba(X_test)[:,0]
        self.predneg['mlp'] = self.clfs['mlpmodel'].predict_proba(X_test)[:,0]
        self.predneg['sgd'] = self.clfs['sgdmodel'].predict_proba(X_test)[:,0]
        self.predneg['xgb'] = self.clfs['xgbmodel'].predict_proba(X_test)[:,0]
        self.predneg['ada'] = self.clfs['adamodel'].predict_proba(X_test)[:,0]
        #self.pred['positive'] = y_test
        #self.predneg['negative'] = [1-y for y in y_test]
        self.pred=pd.DataFrame(self.pred)
        self.pred['vote'] = self.pred.sum(axis=1)
        self.pred['probs'] = self.pred['vote'].apply((lambda x: x/7))
        self.pred['vote'] = self.pred['probs'].apply((lambda x: 1 if x>=self.threshold else 0))

        self.predneg=pd.DataFrame(self.predneg)
        self.predneg['vote'] = self.predneg.sum(axis=1)
        self.predneg['probs'] = self.predneg['vote'].apply((lambda x: x/7))
        self.predneg['vote'] = self.predneg['probs'].apply((lambda x: 1 if x>=self.threshold else 0))

        self.pred['negprob'] = self.predneg['probs'].values

        self.y_pred = self.pred['vote'].values
        self.y_probs = self.pred['probs'].values
        
        self.y_predneg = self.predneg['vote'].values
        self.y_probsneg = self.predneg['probs'].values


        def update_vals(row):
            if row.negprob > 0.5 and row.vote==1:
                row.vote = 0.0
                #print('found 1')
            return row

        self.pred = self.pred.apply(update_vals, axis=1)

        return(self.y_probs)
    
    def getScores_(self):
        return self.auc, self.precision, self.recall

    def score(self, X, y=None, beta=1):
        # gives an F-score with beta=2
        y_pred=np.round(self.predict(X))
        truepos = sum([(lambda t,p: 1 if(t==p and t==1) else 0)(t,p) for (t,p) in zip(y,y_pred)])
        trueneg = sum([(lambda t,p: 1 if(t==p and t==0) else 0)(t,p) for (t,p) in zip(y,y_pred)])
        falsepos = sum([(lambda t,p: 1 if(t!=p and t==0) else 0)(t,p) for (t,p) in zip(y,y_pred)])
        falseneg = sum([(lambda t,p: 1 if(t!=p and t==1) else 0)(t,p) for (t,p) in zip(y,y_pred)])
        self.precision=(truepos/(truepos+falsepos))
        self.recall=(truepos/(truepos+falseneg))
        self.auc = roc_auc_score(y, y_pred)
        self.fbeta = fbeta_score(y,y_pred,beta=beta)
        fbeta = (1+beta**2)*self.recall*self.precision/(beta**2*self.precision+self.recall)
        print('Precision: %.2f' % self.precision)
        print('Recall: %.2f' % self.recall)
        print('AUC: %.2f' % self.auc)
        print('F(%.1f): %.2f' % (beta, self.fbeta))
        return(fbeta)
    
    def plot_submodel_roc_curves(self, y_test, exp_name='',saveFig=False,figPath=''):
        self.roc_curves={}
        f, axarr = plt.subplots(1, 2, figsize=(20, 5))
        axarr[0].set_title('Receiver Operating Charasteristic'+' '+exp_name, fontsize=18)
        axarr[1].set_title('Precision-Recall Curve'+' '+exp_name, fontsize=18)
        for clf in self.classifiers:
            self.roc_curves[clf]={}
            self.roc_curves[clf]['fpr'],self.roc_curves[clf]['tpr'],self.roc_curves[clf]['thr'] =\
                roc_curve(y_test, self.pred[clf])
            self.roc_curves[clf]['auc_score'] = roc_auc_score(y_test, self.pred[clf])
            prec, rec, _ = precision_recall_curve(y_test, self.pred[clf])
            self.roc_curves[clf]['prec'], self.roc_curves[clf]['rec'] = prec, rec
            self.roc_curves[clf]['pr_auc'] = sklearn.metrics.auc(self.roc_curves[clf]['rec'], self.roc_curves[clf]['prec'])
            axarr[0].plot(self.roc_curves[clf]['fpr'],self.roc_curves[clf]['tpr'], \
                     label=clf+', AUC = %.2f' % self.roc_curves[clf]['auc_score'],linestyle=':', alpha=0.8)
            axarr[1].step(self.roc_curves[clf]['rec'],self.roc_curves[clf]['prec'], \
                     label=clf+', AUC = %.2f' % self.roc_curves[clf]['pr_auc'],linestyle=':', alpha=0.8)
            
        self.fpr,self.tpr,self.thr = roc_curve(y_test, self.y_probs)
        self.auc = roc_auc_score(y_test, self.y_probs)
        axarr[0].plot(self.fpr,self.tpr, label='Combined, AUC = %.2f' % self.auc, linestyle='-', alpha=1.0)
        axarr[0].plot([0,1],[0,1], linestyle='--')
        axarr[0].set_xlabel('False Positive Rate', fontsize=15)
        axarr[0].set_ylabel('True Positive Rate', fontsize=15)
        axarr[0].legend(fontsize=12)
        
        self.prec, self.rec, _ = precision_recall_curve(y_test, self.y_probs)
        self.pr_auc = sklearn.metrics.auc(self.rec, self.prec)
        axarr[1].step(self.rec,self.prec, label='Combined, AUC = %.2f' % self.pr_auc, linestyle='-', alpha=1.0)
        axarr[1].set_xlabel('Recall', fontsize=15)
        axarr[1].set_ylabel('Precision', fontsize=15)
        axarr[1].legend(fontsize=12)
        if saveFig:
            figName = 'roc_submodels_'+exp_name+'.png'
            f.savefig(os.path.join(figPath,figName))
        return axarr