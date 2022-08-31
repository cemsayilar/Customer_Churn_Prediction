import numpy as np
import pandas as pd
import scipy.stats as st
import warnings
import joblib
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from datetime import date
#### RFM-CRM-CLTV
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
#### Recomendation Systems
#### Measurement
from statsmodels.stats.proportion import proportions_ztest
#### Feature Engineering
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
#### Machine Learning
from sklearn.metrics import mean_squared_error, mean_absolute_error
### Logistic Regression
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
### KNN
from sklearn.neighbors import KNeighborsClassifier
#### Tree Methods
### CART
#pip install pydotplus
#pip install skompiler
#pip install astor
#pip install joblib
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
# conda install graphviz ## For tree graph!
### GBM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
#from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import graphviz
### K-Means Clustering # pip install yellowbrick
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import cross_val_score, GridSearchCV
### Pipeline
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from Private.helpers import *

# Telco Churn
# Business Problem
# Company wants to build a machine learning model to predict customers that may will 'churn'.

df = pd.read_csv('/Users/buraksayilar/Desktop/machine_learning/ODEV/HW_II_III_IV/Telco_Churn/Telco-Customer-Churn.csv')

################################################
# 1. Exploratory Data Analysis
################################################
check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in cat_cols:
    cat_summary(df, col)

check_outlier(df, 'tenure')
check_outlier(df, 'MonthlyCharges')
# No outliers.

df['Churn'] = np.where(df['Churn'] == 'Yes', 1,0)

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.dropna(inplace=True)

df.isnull().value_counts()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])




######################################################
# 3. Models
######################################################
X = df.drop(['Churn_1', 'customerID'], axis=1)
y = df['Churn_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

# The data is imbalanced. Therefor I use smote method to oversample the data

oversample = SMOTE(random_state=99, k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote
# Normally, Base Models should be in Research section.
# I should determine one or more suitable model for my problem. Then I should conduct Hyperparameter Optimisations.
# After optimisation, I will choose 3 model to conduct my research.

def ml_classification(X_train, X_test, y_train, y_test):
        accuracy, f1, auc, = [], [], []

        random_state = 42

        ##classifiers
        classifiers = []
        classifiers.append(DecisionTreeClassifier(random_state=random_state))
        classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)))
        classifiers.append(RandomForestClassifier(random_state=random_state))
        classifiers.append(GradientBoostingClassifier(random_state=random_state))
        classifiers.append(XGBClassifier(random_state=random_state))
        #classifiers.append(LGBMClassifier(random_state=random_state))
        classifiers.append(CatBoostClassifier(random_state=random_state, verbose=False))

        for classifier in classifiers:
            # classifier and fitting
            clf = classifier
            clf.fit(X_train, y_train)

            # predictions
            y_preds = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)

            # metrics
            accuracy.append((cross_validate(classifier, X_train, y_train, cv=3, scoring='accuracy')['test_score'].mean())* 100)
            f1.append((cross_validate(classifier, X_train, y_train, cv=3, scoring='f1')['test_score'].mean()) * 100)
            auc.append((cross_validate(classifier, X_train, y_train, cv=3, scoring='roc_auc')['test_score'].mean()) * 100)

        results_df = pd.DataFrame({"Accuracy Score": accuracy,
                                   "f1 Score": f1, "AUC Score": auc,
                                   "ML Models": ["DecisionTree", "AdaBoost",
                                                 "RandomForest", "GradientBoosting",
                                                 "XGBoost", "CatBoost"]})

        results = (results_df.sort_values(by=['f1 Score'], ascending=False).reset_index(drop=True))

        return results
def ml_classification(X_train, X_test, y_train, y_test):
    accuracy, f1, auc, = [], [], []

    random_state = 42

    ##classifiers
    classifiers = []
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(XGBClassifier(random_state=random_state))
    #classifiers.append(LGBMClassifier(random_state=random_state))
    classifiers.append(CatBoostClassifier(random_state=random_state, verbose=False))

    for classifier in classifiers:
        # classifier and fitting
        clf = classifier
        clf.fit(X_train, y_train)

        # predictions
        y_preds = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)

        # metrics
        accuracy.append(((accuracy_score(y_test, y_preds))) * 100)
        f1.append(((f1_score(y_test, y_preds))) * 100)
        auc.append(((roc_auc_score(y_test, y_probs[:, 1]))) * 100)

    results_df = pd.DataFrame({"Accuracy Score": accuracy,
                               "f1 Score": f1, "AUC Score": auc,
                               "ML Models": ["DecisionTree", "AdaBoost",
                                             "RandomForest", "GradientBoosting",
                                             "XGBoost", "CatBoost"]})

    results = (results_df.sort_values(by=['f1 Score'], ascending=False)
               .reset_index(drop=True))

    return results
ml_classification(X_train, X_test, y_train, y_test)

# It seams Gradient Boosting, Random Forest and XGBoost are the strongest methods for my problem.
# Lets see the feeature importance.

gbm_model = GradientBoostingClassifier()
gbm_model = gbm_model.fit(X_train, y_train)
gbm_pred = gbm_model.predict(X_test)
cv_results = cross_validate(gbm_model, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Feature Importance
feature_imp = pd.Series(gbm_model.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)
sns.barplot(x= feature_imp[0:10]*100, y = feature_imp.index[0:10])
plt.xlabel("Variable Scores")
plt.ylabel("Variables")
plt.title("Feature Importance")
plt.show()

gbm_model_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
gbm_final = gbm_model.set_params(**gbm_model_best_grid.best_params_, random_state=17, ).fit(X_train, y_train)

cv_results = cross_validate(gbm_final, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.82  --->  0.83
cv_results['test_f1'].mean()
# 0.82  --->  0.81
cv_results['test_roc_auc'].mean()
# 0.9120  --->  0.9137