
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from imblearn.metrics import geometric_mean_score
import warnings
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC 
import xgboost as xgb
from vecstack import stacking
from scipy import stats
import os



# Import DataSet
dt = pd.read_csv(
    r'C:\Users\Lenovo\Desktop\Deployment HD\heart_statlog_cleveland_hungary_final.csv')
dt.head()




# Preprocessing the DataSet
# renaming features to proper name
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 
              'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope','target']




# converting features to categorical features 

#dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
#dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
#dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
#dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'



#dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
#dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
#dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'


#dt['st_slope'][dt['st_slope'] == 0] = 'normal'
#dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
#dt['st_slope'][dt['st_slope'] == 2] = 'flat'
#dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

#dt["sex"] = dt.sex.apply(lambda  x:'male' if x==1 else 'female')



## Checking missing entries in the dataset columnwise
dt.isna().sum()




# Outlier Detection & Removal

# filtering numeric features as age , resting bp, cholestrol and max heart rate achieved has outliers as per EDA

dt_numeric = dt[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved']]

dt_numeric.head()




# calculating zscore of numeric columns in the dataset

from scipy import stats
z = np.abs(stats.zscore(dt_numeric))
print(z)




# Defining threshold for filtering outliers 

threshold = 3
print(np.where(z > 3))




#filtering outliers retaining only those data points which are below threshhold

dt = dt[(z < 3).all(axis=1)]


dt.shape




## encoding categorical variables

#dt = pd.get_dummies(dt)

#dt.head()

# converting features to categorical features 

#dt['chest_pain_type'][dt['chest_pain_type'] == 'atypical angina'] = 2
#dt['chest_pain_type'][dt['chest_pain_type'] == 'non-anginal pain'] = 3
#dt['chest_pain_type'][dt['chest_pain_type'] == 'asymptomatic'] = 4



#dt['rest_ecg'][dt['rest_ecg'] == 'normal'] = 0
#dt['rest_ecg'][dt['rest_ecg'] == 'ST-T wave abnormality'] = 1
#dt['rest_ecg'][dt['rest_ecg'] == 'left ventricular hypertrophy'] = 2


#dt['st_slope'][dt['st_slope'] == 'normal'] = 0
#dt['st_slope'][dt['st_slope'] == 'upsloping'] = 1
#dt['st_slope'][dt['st_slope'] == 'flat'] = 2
#dt['st_slope'][dt['st_slope'] == 'downsloping'] = 3

#dt["sex"] = dt.sex.apply(lambda  x:1 if x=='male' else 0)





#Train Test Split 

# segregating dataset into features i.e., X and target variables i.e., y

X = dt.drop(['target'],axis=1)
y = dt['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y, 
    test_size=0.2,
    shuffle=True, 
    random_state=5)




print('------------Training Set------------------')
print(X_train.shape)
print(y_train.shape)

print('------------Test Set------------------')
print(X_test.shape)
print(y_test.shape)



# Feature Normalization

scaler = MinMaxScaler()
X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.fit_transform(X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])
X_train.head()




X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.transform(X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])
X_test.head()




X_train.tail()




# Cross Validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import xgboost as xgb
# function initializing baseline machine learning models

def GetBasedModel():
    basedModels = []
    basedModels.append(('LR_L2'   , LogisticRegression(penalty='l2')))
    basedModels.append(('KNN9'  , KNeighborsClassifier(9)))
    basedModels.append(('KNN11'  , KNeighborsClassifier(11)))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear',gamma='auto',probability=True)))
    basedModels.append(('SVM RBF'  , SVC(kernel='rbf',gamma='auto',probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier(n_estimators=100,max_features='sqrt')))
    basedModels.append(('RF_Ent100'   , RandomForestClassifier(criterion='entropy',n_estimators=100)))
    basedModels.append(('RF_Gini100'   , RandomForestClassifier(criterion='gini',n_estimators=100)))
    basedModels.append(('ET100'   , ExtraTreesClassifier(n_estimators= 100)))
    basedModels.append(('MLP', MLPClassifier()))
    basedModels.append(('XGB_500', xgb.XGBClassifier(n_estimators= 500, eval_metric='logloss')))
    basedModels.append(('XGB_100', xgb.XGBClassifier(n_estimators= 100, eval_metric='logloss')))
    return basedModels




# function for performing 15-fold cross validation of all the baseline models
from sklearn.model_selection import KFold

def BasedLine2(X_train, y_train,models):
    # Test options and evaluation metric

    scoring = 'accuracy'
    seed = 7
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=15,shuffle=True,random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
        print(msg)
         
        
    return results,msg




models = GetBasedModel()
names,results = BasedLine2(X_train, y_train,models)




# Random Forest Classifier (criterion = 'entropy')
rf_ent = RandomForestClassifier(criterion='entropy',n_estimators=100)
rf_ent.fit(X_train, y_train)
y_pred_rfe = rf_ent.predict(X_test)




# Multi Layer Perceptron
mlp = MLPClassifier()
mlp.fit(X_train,y_train)
y_pred_mlp = mlp.predict(X_test)




# K nearest neighbour (n=9)
knn = KNeighborsClassifier(9)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)




# Extra Tree Classifier (n_estimators=500)
et_500 = ExtraTreesClassifier(n_estimators= 500)
et_500.fit(X_train,y_train)
y_pred_et500 = et_500.predict(X_test)




# XGBoost (n_estimators=500)
xgb = xgb.XGBClassifier(n_estimators= 500, eval_metric='logloss')
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)




# Support Vector Classifier (kernel='linear')
svc = SVC(kernel='linear',gamma='auto',probability=True)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)



# Adaboost Classifier
ada = AdaBoostClassifier()
ada.fit(X_train,y_train)
y_pred_ada = ada.predict(X_test)




# decision Tree Classifier (CART)
decc = DecisionTreeClassifier()
decc.fit(X_train,y_train)
y_pred_decc = decc.predict(X_test)




# gradient boosting machine
gbm = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gbm.fit(X_train,y_train)
y_pred_gbm = gbm.predict(X_test)




# Model Selection

import xgboost as xgboost
# selecting list of top performing models to be used in stacked ensemble method
models = [
    RandomForestClassifier(criterion='entropy',n_estimators=100),
    MLPClassifier(),
    RandomForestClassifier(criterion='gini',n_estimators=100),
    KNeighborsClassifier(9),
    ExtraTreesClassifier(n_estimators= 500),
    ExtraTreesClassifier(n_estimators= 100),
    xgboost.XGBClassifier(n_estimators= 100, eval_metric='logloss'),
    xgboost.XGBClassifier(n_estimators= 500, eval_metric='logloss'), 
    SVC(kernel='linear',gamma='auto',probability=True),
    AdaBoostClassifier(),
    DecisionTreeClassifier(),
    GradientBoostingClassifier(n_estimators=100,max_features='sqrt'),
]




# Stacked Ensemble
S_train, S_test = stacking(models,                   
                           X_train, y_train, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=5, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)




model = RandomForestClassifier(criterion='entropy',n_estimators=100)
#import xgboost as xgb
#model = xgb.XGBClassifier(n_estimators= 500, eval_metric='logloss')    

model = model.fit(X_train, y_train)

"""
y_pred = model.predict(X_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))

#Randomized Search CV

#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features =  ['auto', 'sqrt']
# max number of leaves in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Min number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Min number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = model, param_distributions= random_grid, 
                               scoring='neg_mean_squared_error', n_iter=10, 
                               cv= 5, verbose=2, random_state=42, n_jobs= 1 )

rf_random.fit(S_train, y_train)

rf_random.best_params_

prediction= rf_random.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, prediction))
"""
import pickle

#open a file , where dump the model
file = open('hd_predict.pkl', 'wb')
pickle.dump(model, file)

model1= open('hd_predict.pkl', 'rb')
forest = pickle.load(model1)
y_pred = forest.predict(X_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
""""""

# building a prediction system

input_data = (48,0,4,138,214,0,0,108,1,1.5,2)

# #53,124,243,0,122,1,2,1,1,0,0,0,0,1,1,0
# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = forest.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')