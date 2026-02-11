import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest, SelectPercentile
def chi_select(X, y, k=None, percentile=None): 
    ## Univariate feature selection
    ### only for non-negative variables
    if k:       # top k
        percentile = None
        X_chi = SelectKBest(chi2, k=300).fit_transform(X, y)
    if percentile:  # top percentile
        X_chi = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    # chi2 Chi-squared values and corresponding p-values between features and the target
    # chi, p = chi2(X_fsvar,y)
    return X_chi 


###bmethod 1
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, f_regression
def UnivariateFeatureSelection(X, y, n_features=1):
    ''' 
    Compute the ANOVA F-value for the provided sample.
    ''' 
    X_train, y_train = X, y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    univariate = f_classif(X_train, y_train)
    ## Capture P values in a series
    univariate = pd.Series(univariate[1])
    univariate.index = X_train.columns
    univariate.sort_values(ascending=False, inplace=True)
    # ## Plot the P values
    # univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))
    ## Select K best Features
    k_best_features = SelectKBest(f_classif, k=n_features).fit(X_train.fillna(0), y_train) 
    # k_best_features = SelectPercentile(f_classif, percentile=10).fit(X_train.fillna(0), y_train)
    # print( X_train.columns[k_best_features.get_support()] )
    # selected_features = X_train.columns[k_best_features.get_support()]
    # # print( X_train.shape ) 
    # X_train = k_best_features.transform(X_train.fillna(0))

    feats_score = k_best_features.scores_ 
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()
    return X_train_np, selected_features_sort_large_small 

##bmethod 2
from sklearn.feature_selection import mutual_info_classif
def mutual_info_selection(X, y, n_features=10): 
    # Calculate Mutual Information between each feature and the target

    X_train, y_train = X, y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    mutual_info = mutual_info_classif(X_train.fillna(0), y_train, discrete_features=False)
    ## Create Feature Target Mutual Information Series
    mi_series = pd.Series(mutual_info)
    mi_series.index = X_train.columns
    mi_series.sort_values(ascending=False) 
    # mi_series.sort_values(ascending=False).plot.bar(figsize=(20,8))
    # # Select K best features
    k_best_features = SelectKBest(mutual_info_classif, k=n_features).fit(X_train.fillna(0), y_train) 
    # k_best_features = SelectPercentile(mutual_info_classif, percentile=10).fit(X_train.fillna(0), y_train)
    # print( X_train.columns[k_best_features.get_support()] )
    # selected_features = X_train.columns[k_best_features.get_support()]
    # X_train = k_best_features.transform(X_train.fillna(0)) 

    feats_score = k_best_features.scores_ 
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()
    return X_train_np, selected_features_sort_large_small 


###bmethod 3
from sklearn.feature_selection import VarianceThreshold
def varianceThreshold_rm(X, var_thresh=0): 
    ## method1
    selector = VarianceThreshold(threshold = var_thresh)
    selector.fit_transform(X)
    aa_name = selector.get_support()
    selected_features = selector.get_feature_names_out()
    aa_select = X.iloc[:,aa_name]
    # ## method2 
    # var = feature_frame.var()
    # var_thresh = var_thresh
    # var_select = var[var>var_thresh]
    # aa_select = feature_frame[var_select.index]
    return aa_select, selected_features 

import mrmr    ###pip install mrmr_selection
from mrmr import mrmr_classif
def mrmr_selection(X, y, n_features=1):
    # select top 10 features using mRMR
    X_train, y_train = X, y
    selected_features, feats_score, _ = mrmr_classif(X=X_train, y=y_train, K=n_features, n_jobs=10, return_scores=True, show_progress=False) 
    # X = X[selected_features] 
    # X = X.to_numpy()
    # selected_features are feature names
    feats_score = feats_score.to_numpy()
    feats_score_sort_large_small = feats_score.argsort()[-n_features:][::-1] 

    selected_features_sort_large_small = X_train.columns[feats_score_sort_large_small]        # feature name
    X_train_np = X_train[selected_features_sort_large_small].fillna(0).to_numpy()             # feature value
    return X_train_np, selected_features_sort_large_small

from scipy import stats
def ttest_selection(X, y, n_features=1, sig=0.05 ):
    feats_name = X.columns.tolist()
    X_train = X.to_numpy()
    y_train = np.squeeze( y )     
    X_train_0 = X_train[y_train==0,:]
    X_train_1 = X_train[y_train==1,:]
    pv_list = []
    for i in range(X_train_0.shape[1]): 
        t_stat, pv = stats.ttest_ind(X_train_0[:, i], X_train_1[:, i] ) 
        pv_list.append(pv ) 
    indices = np.argsort(pv_list).tolist()
    sorted_pv_list = [pv_list[index] for index in indices] 
    sorted_feats_name = [feats_name[index] for index in indices] 
    selected_features = sorted_feats_name[:n_features] 
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features ## sorted_pv_list, sorted_feats_name

##Wilcoxon rank-sum statistic
def ranksums_selection(X, y, n_features=6, sig=0.05 ):
    feats_name = X.columns.tolist()
    X_train = X.to_numpy()
    y_train = np.squeeze( y )     
    X_train_0 = X_train[y_train==0,:]
    X_train_1 = X_train[y_train==1,:]
    pv_list = []
    for i in range(X_train_0.shape[1]): 
        stat, pv = stats.ranksums(X_train_0[:, i], X_train_1[:, i])
        pv_list.append(pv)
    indices = np.argsort(pv_list).tolist()
    sorted_pv_list = [pv_list[index] for index in indices] 
    sorted_feats_name = [feats_name[index] for index in indices] 
    selected_features = sorted_feats_name[:n_features] 
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features ## sorted_pv_list, sorted_feats_name  

from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.svm import SVR, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Recursive Feature Elimination
# def rfe_selection(X, y, n_features=50):
#     # estimator = SVR(kernal='linear')  ##need to change
#     estimator = SVC(kernel="linear")
#     selector = RFE(estimator, n_features_to_select=n_features, step=1).fit(X, y)
#     # selector = RFECV( estimator=estimator, step=1, cv=StratifiedKFold(2), scoring="accuracy", min_features_to_select=1, )
#     selected_features = selector.support_
#     print(selected_features)
#     selected_ranking = selector.ranking_
#     # print("Optimal number of features : %d" % selector.n_features_)  ###FOR RFECV
#     X_selected = X.iloc[:, selected_features]
#     return X_selected
def rfe_selection(X, y, n_features=50):
    # estimator = SVR(kernal='linear')  ##need to change
    feature_names = X.columns
    estimator = SVC(kernel="linear")
    # estimator = LogisticRegression()    # poorer than SVC linear
    selector = RFE(estimator, n_features_to_select=n_features, step=1).fit(X, y)
    # selector = RFECV( estimator=estimator, step=1, cv=StratifiedKFold(2), scoring="accuracy", min_features_to_select=1, )
    selected_features = selector.support_         # bool
    # selected_ranking = selector.ranking_
    selected_feature_names = feature_names[selected_features]
    # print("Optimal number of features : %d" % selector.n_features_)  ###FOR RFECV
    X_selected = X[selected_feature_names]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_feature_names


def elastic_net_slectiion(X, y, n_features=50):
    feature_names = X.columns
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1)   # high l1_ratio performance bad

    lr.fit(X, y)
    coefficients = lr.coef_
    coef_abs = np.abs(coefficients)
    indices = np.argsort(coef_abs)[0][::-1]
    selected_indices = indices[:n_features]
    selected_feature_names = feature_names[selected_indices]
    X_selected = X[selected_feature_names]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_feature_names



from sklearn.ensemble import RandomForestClassifier
def random_forest_slectiion(X, y, n_features=50):
    feature_names = X.columns
    rf = RandomForestClassifier(n_estimators=100)           # 100 enough
    rf.fit(X, y)

    importances = rf.feature_importances_

    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_features]
    selected_features = feature_names[top_indices]
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features

import xgboost as xgb
def XGBoost_slectiion(X, y, n_features=50):
    feature_names = X.columns
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)

    importances = xgb_model.feature_importances_
    # Ranking of importance, selecting the top N features
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_features]
    selected_features = feature_names[top_indices]
    X_selected = X[selected_features]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_features


from sklearn.linear_model import Lasso
def lasso_feature_selection(X, y, n_features=50):
    feature_names = X.columns
    lasso = Lasso(alpha=0.0001)         # moderately loose
    lasso.fit(X, y)
    coefficients = lasso.coef_

    non_zero_indices = np.where(coefficients != 0)[0]
    if len(non_zero_indices) < n_features:
        selected_indices = non_zero_indices
    else:
        coef_abs = np.abs(coefficients)
        indices = np.argsort(coef_abs)[::-1]
        selected_indices = indices[:n_features]
    selected_feature_names = feature_names[selected_indices]

    X_selected = X[selected_feature_names]
    X_selected = X_selected.to_numpy()
    return X_selected, selected_feature_names




from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
def SelectFromModel_(X, y):
    RFC_ = RFC(n_estimators=10)
    ##### print(X_embedded.shape)
    
    RFC_.fit(X,y).feature_importances_
    threshold = np.linspace(0,(RFC_.fit(X,y).feature_importances_).max(),20)
    score = []
    for i in threshold:
        X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(X,y)
        once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
        score.append(once)
    plt.plot(threshold,score)
    plt.show()
    X_embedded = SelectFromModel(RFC_,threshold=0.00067).fit_transform(X,y)
    X_embedded.shape
    print(X_embedded.shape)
    print(cross_val_score(RFC_,X_embedded,y,cv=5).mean()) 
    # from sklearn.svm import LinearSVC
    # from sklearn.datasets import load_iris
    # from sklearn.feature_selection import SelectFromModel
    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_new = model.transform(X)
    # X_new.shape 
