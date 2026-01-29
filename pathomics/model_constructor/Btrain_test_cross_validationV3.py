from operator import index
from joblib import dump
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ranksums as rs
import random
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from Bclassifiers import QDA_, LDA_, RandomForestC_, SGDClassifier_, DecisionTree_, \
                        KNeighbors_, LinearSVC_, SVC_rbf_, GaussianProcess_, \
                            MLPClassifier_, AdaBoostClassifier_, GaussianNB_, Ridge_
from Bfeatures_selection import (UnivariateFeatureSelection, mutual_info_selection, mrmr_selection, \
                            ttest_selection, ranksums_selection, varianceThreshold_rm, rfe_selection, chi_select,
                            elastic_net_slectiion, random_forest_slectiion, XGBoost_slectiion, lasso_feature_selection)
import time 
from datetime import datetime
from multiprocessing import Pool, Manager
from itertools import repeat
import os
import argparse
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
import glob
import random
from lifelines import KaplanMeierFitter, CoxPHFitter
from matplotlib.patches import RegularPolygon
from scipy.stats import ranksums
import scipy.stats as stats
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc as AUROC
import math
from pathlib import Path

def remove_correlated_features(df, corr_threshold=0.8):
    """
    Remove correlated features such that only features with a pearson 
    correlation coeffcient of less than 'corr_threshold' remain in the dataframe.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with features.
    non_feature_columns : list of strings
        List of columns which do not contain features.
    corr_threshold: float
        If two features have a pearson correlation index above the 
        corr_threshold, one of the features will be dropped from the 
        dataframe.
   
    Returns
    -------
    df : pandas dataframe
        Dataframe without heavily correlated feautures (less than 'corr_threshold').
    dropped_features : list of strings
        List of features that have been dropped.
    """
    
    print("Remove correlated features...")

    # Check corr_threshold
    if type(corr_threshold) is float:
        if 0 < corr_threshold and corr_threshold < 1:
            print("Correlation threshold:", corr_threshold)
        else:
            print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
            return -1
    else:
        print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
        return -1
    
    # Get names and number of all features (columns)
    all_features = df.columns.to_list()
    n_features = df.shape[1]

    ## normalization 
    ifMinMaxNorm=False 
    ifZScore=False
    if ifMinMaxNorm: 
        ##min-max normalization
        df=(df-df.min())/(df.max()-df.min())
    elif ifZScore: 
        ##Z-score normalization 
        df=(df-df.mean())/df.std() 

    # Compute correlation matrix
    corr = df.corr(method='spearman') ##pearson
    corr = corr.abs()

    # Keep only correlation values in the upper traingle matrix
    triu = np.triu(np.ones(corr.shape), k=1)
    triu = triu.astype(bool)    ################# np.bool
    corr = corr.where(triu)
    
    # Select columns which will be dropped from dataframe
    cols_to_drop = [column for column in corr.columns if any(corr[column] > corr_threshold)]

    n_cols_to_drop = len(cols_to_drop)
    p_cols_to_drop = 100 * (float(n_cols_to_drop) / float(n_features))
    p_cols_to_drop = np.round(p_cols_to_drop, decimals=1)
    print("Drop", n_cols_to_drop, "/", n_features, " features (", p_cols_to_drop, "%).")

    # Drop colums
    df = df.drop(cols_to_drop, axis=1)
    
    # Find names of features which have been dropped
    uncorrelated_features = df.columns.to_list()
    dropped_features = list(set(all_features) - set(uncorrelated_features))
    
    return df, dropped_features



# Insert the AUC values from df_auc_table directly
def ROC_Chart_set(n_feat_sel, nclass, all_roc_fpr, all_roc_tpr,
                  all_roc_fpr_std, all_roc_tpr_std, df_auc_table, save_path,
                  list_feats_selection_args, list_classifers_args, max_row_ind, max_col_ind):
    fig, axes = plt.subplots(n_feat_sel, nclass, figsize=(20, 14))
    b = 0
    for i in range(n_feat_sel):
        for j in range(nclass):
            # get ROC curve data of the current position
            roc_fpr = all_roc_fpr.iloc[:, b]
            roc_tpr = all_roc_tpr.iloc[:, b]
            roc_fpr_fill_std = all_roc_fpr_std.iloc[:, b]
            roc_tpr_fill_std = all_roc_tpr_std.iloc[:, b]
            auc_value = float(df_auc_table.iloc[i, j].split('/')[0])

            # Sort fpr and tpr to ensure they are monotonically increasing
            sort_indices = np.argsort(roc_fpr)
            roc_fpr_sorted = roc_fpr.iloc[sort_indices]
            roc_tpr_sorted = roc_tpr.iloc[sort_indices]
            roc_tpr_fill_std_sorted = roc_tpr_fill_std.iloc[sort_indices]

            # plot ROC curve
            axes[i, j].plot(roc_fpr_sorted, roc_tpr_sorted, label='ROC', color='darkorange', lw=2)
            axes[i, j].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[i, j].fill_between(roc_fpr_sorted, roc_tpr_sorted - roc_tpr_fill_std_sorted,
                                    roc_tpr_sorted + roc_tpr_fill_std_sorted, color='skyblue', alpha=0.5, label='Std')

            axes[i, j].text(0.95, 0.05, "AUC={:.3f}".format(auc_value), transform=axes[i, j].transAxes, fontsize=8,
                            va='bottom', ha='right')
            axes[i, j].set_title(f'({list_feats_selection_args[i]}, {list_classifers_args[j]})', fontsize=11)
            axes[i, j].set_xlabel('False Positive Rate')
            axes[i, j].set_ylabel('True Positive Rate')
            axes[i, j].legend(fontsize=6)
            if i == max_row_ind and j == max_col_ind:
                axes[i, j].set_facecolor('#E7EFFA')
            b += 1
    plt.suptitle('ROCs of different feature selection methods and classifiers combination matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{save_path}/ROC_curve.png')
    plt.close()



def violin_plot(n_row, n_col, topk_features, features_matrix,
                labels, feats_selection_name, classifiers_name,
                save_path) -> None:
    """
    Plot violin plot of top features in every combination
    :param n_row: The number of rows
    :param n_col: The number of columns
    :param topk_features: The names of topk_features
    :param features_matrix: Feature matrix
    :param labels: Labels in the same order as the feature matrix
    :param feats_selection_name: The name of feature selection
    :param classifiers_name: The name of feature classifier
    :param save_path: Save path
    """
    fig, axes = plt.subplots(n_row, n_col, figsize=(40, 21))
    row, col = 0, 0
    for feature in topk_features:
        feature_values = features_matrix[feature]
        # label_values = matrix_M['label'].replace({0: 'Long_term_survial', 1: 'Short_term_survial'})
        label_values = labels.replace({0: 'Long_term', 1: 'Short_term'})
        label_values = label_values.squeeze()

        feature_scores_df = pd.DataFrame({'Feature': feature_values, 'Label': label_values})

        # plot violin plot
        sns.violinplot(x='Label', y='Feature', data=feature_scores_df, inner='quartile', palette='Set1',
                       ax=axes[row, col])  # , hue='Label', palette='Set1')
        sns.stripplot(x='Label', y='Feature', data=feature_scores_df, jitter=True, color='red', size=4, palette='Set3',
                      ax=axes[row, col])
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('Feature Value', fontsize=35)
        axes[row, col].set_title(f'{feature}', fontsize=35)
        axes[row, col].tick_params(axis='x', labelsize=35)
        axes[row, col].tick_params(axis='y', labelsize=35)

        # calculate p value
        feature_scores_0 = feature_scores_df[feature_scores_df['Label'] == 'Long_term']['Feature']
        feature_scores_1 = feature_scores_df[feature_scores_df['Label'] == 'Short_term']['Feature']
        _, p = ranksums(feature_scores_0, feature_scores_1)
        if p < 0.001:
            p = 0.001
            axes[row, col].text(0.5, 0.8, f'p< {p:.3f}', ha='center', va='center', fontsize=28,
                                transform=axes[row, col].transAxes)
        else:
            axes[row, col].text(0.5, 0.8, f'p= {p:.3f}', ha='center', va='center', fontsize=28,
                                transform=axes[row, col].transAxes)
        # ax.text(0.5, 0.8, f'p= {p:.3f}', ha='center', va='center', fontsize=18, transform=ax.transAxes)

        col += 1
        if col == 3:
            row += 1
            col = 0
    plt.tight_layout()
    folder_time1 = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    plt.savefig(
        f'{save_path}/{feats_selection_name}_{classifiers_name}_{folder_time1}_violinplots.png')
    plt.close()




def horizontal_bar_chart(topk_features, topk_scores_percentage,
                         feats_selection_name, classifiers_name,
                         save_path) -> None:
    """
    Plot horizontal bar plot of top features in every combination
    :param topk_features: The names of topk_features
    :param topk_scores_percentage: The percentage of topk_features scores
    :param feats_selection_name: The name of feature selection
    :param classifiers_name: The name of classifier
    :param save_path: Save path
    """

    plt.figure(figsize=(65, 50))
    plt.subplots_adjust(left=0.4)
    bars = plt.barh(topk_features[::-1], topk_scores_percentage[::-1])
    plt.ylabel('Selected Features', fontsize=75)
    plt.xlabel('Selection Probability (%)', fontsize=75)
    # plt.title(
    #         f'Top 12 Selected Features - {feats_selection_name}, {classifiers_name}', fontsize=75)
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height() / 2, f'{xval:.2f}%', va='center', color='red',
                 fontweight='bold', fontsize=54)

    plt.xticks(fontsize=75)
    plt.yticks(fontsize=60)

    folder_time1 = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    plt.savefig(
        f'{save_path}/{feats_selection_name}_{classifiers_name}_{folder_time1}.png')


def calculate_layout(n):
    """
    calculate the layout of n graphs

    """
    if n <= 0:
        return 0, 0

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # more like a square
    if rows > cols and cols > 1:
        rows, cols = cols, rows
        if rows * cols < n:
            rows += 1

    return rows, cols

def voting(pred_list, pred_proba_list, vote_type='soft', weights=None):
    if vote_type not in ("soft", "hard"):
        raise ValueError(
            f"Voting must be 'soft' or 'hard'; got (voting={vote_type!r})"
        )

    if weights is None: 
        weights = [1, ]*len(pred_list) 

    # np.asarray(pred_list).T
    pred_list = np.asarray(pred_list)
    pred_proba_list = np.asarray(pred_proba_list) 

    pred_proba_avg = np.average(
        pred_proba_list, axis=0, weights=weights
    )
    if vote_type == "soft": 

        maj = np.argmax(pred_proba_avg, axis=1)

    else:  # 'hard' voting
        predictions = pred_list.T
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=1,
            arr=predictions,
        )
    return maj, pred_proba_avg


def imBalanced_MES(estimator, X, y, X_test, y_test, params_dict=None): 
    '''Multiexpert systems (MES)  
    considers the imbalanced classes distribution'''
    percent_1 = np.sum(y==1) / y.shape[0]  
    ind_1 = np.where(y==1)
    ind_0 = np.where(y==0)
    X_1, X_0 = X[ind_1[0], :], X[ind_0[0], :] 
    y_1, y_0 = y[ind_1[0], :], y[ind_0[0], :] 
    len_1 = len(ind_1[0]) 
    len_0 = len(ind_0[0])
    if len_1 > len_0: 
        ind_min, ind_max = len_0, len_1 
        split_points = list(range(0, ind_max, ind_min))
        if ( ind_max - split_points[-1]) / ind_min < 0.5:
            split_points = split_points[:-1] 
        y_test_pred_list = []
        y_test_score_pred_list = []
        for i in range(len(split_points)): 
            X_1_equal = X_1[split_points[i]: split_points[i]+ind_min, :]
            y_1_equal = y_1[split_points[i]: split_points[i]+ind_min, :]
            X_train, y_train = np.concatenate([X_1_equal, X_0], axis=0), np.concatenate([y_1_equal, y_0], axis=0) 
            if params_dict is not None: 
                y_test_pred, y_test_score_pred = estimator(X_train, y_train, params_dict).predict(X_test) 
            else: 
                y_test_pred, y_test_score_pred = estimator(X_train, y_train).predict(X_test) 
            y_test_pred_list.append(y_test_pred ) 
            y_test_score_pred_list.append(y_test_score_pred ) 
    else: 
        ind_min, ind_max = len_1, len_0  
        split_points = list(range(0, ind_max, ind_min) ) 
        if ( ind_max - split_points[-1] ) / ind_min < 0.5: 
            split_points = split_points[:-1] 
        y_test_pred_list = []
        y_test_score_pred_list = []
        for i in range(len(split_points)): 
            X_0_equal = X_0[split_points[i]: split_points[i]+ind_min, : ] 
            y_0_equal = y_0[split_points[i]: split_points[i]+ind_min, : ] 
            X_train, y_train = np.concatenate([X_0_equal, X_1], axis=0), np.concatenate([y_0_equal, y_1], axis=0) 
            # perm = np.arange(X_train.shape[0])
            # np.random.shuffle(perm) 
            # X_train, y_train = X_train[perm, :], y_train[perm, :]
            if params_dict is not None: 
                y_test_pred, y_test_score_pred = estimator(X_train, y_train, params_dict).predict(X_test) 
            else: 
                y_test_pred, y_test_score_pred = estimator(X_train, y_train).predict(X_test) 
            # print( classifier_measures(y_test, y_test_pred, y_test_score_pred) )
            y_test_pred_list.append(y_test_pred ) 
            y_test_score_pred_list.append(y_test_score_pred ) 
    # y_test_score_pred_vote = voting(y_test_pred_list, y_test_score_pred_list, vote_type='soft')
    y_test_pred_vote, y_test_score_pred_vote = voting(y_test_pred_list, y_test_score_pred_list, vote_type='soft')
    
    return y_test_pred_vote, y_test_score_pred_vote 


def findIndexOfFeatures(fullFeatures, selectedFeatrues): 
    ind_list = [fullFeatures.index(selectedFeatrues[i]) for i in range(len(selectedFeatrues))]
    return ind_list 

def findTopKFeatures(fullFeatures, features_scores, topk=10): 
    # get topk feature names
    features_scores_sort_ind = np.argsort(features_scores)[::-1]
    topkFeatures = [fullFeatures[features_scores_sort_ind[i]] for i in range(topk)]

    return topkFeatures




# def findTopKFeatures_rate(fullFeatures, features_scores, topk=10):
#     # get topk feature names
#     features_scores_sort_ind = np.argsort(features_scores)[::-1]
#     # 获取fullFeatures中以'firstorder'，'glcm'和'glrlm'开头的特征的索引
#     firstorder_indices = [i for i, feature in enumerate(fullFeatures) if feature.startswith('firstorder')]
#     glcm_indices = [i for i, feature in enumerate(fullFeatures) if feature.startswith('glcm')]
#     glrlm_indices = [i for i, feature in enumerate(fullFeatures) if feature.startswith('glrlm')]
#
#     获取每个大类特征索引对应的分数
#     firstorder_scores = [features_scores[i] for i in firstorder_indices]
#     glcm_scores = [features_scores[i] for i in glcm_indices]
#     glrlm_scores = [features_scores[i] for i in glrlm_indices]
#     # 对分数列表进行求和，获得'firstorder'，'glcm'和'glrlm'每一个大类被选择的次数
#     sum_firstorder_scores = np.sum(firstorder_scores)
#     sum_glcm_scores = np.sum(glcm_scores)
#     sum_glrlm_scores = np.sum(glrlm_scores)
#     # 建立“比例”列表，将每一个top特征占其所属大类特征的比例save
#     firstorder_rate_list = []
#     glcm_rate_list = []
#     glrlm_rate_list = []
#     # 遍历topk特征，计算每一个特征的score在其所属大类特征中所占比例
#     for i in range(topk):
#         one_of_topkFeatures = fullFeatures[features_scores_sort_ind[i]]
#         if one_of_topkFeatures.startswith('firstorder'):
#             one_of_topkFeatures_score = features_scores[features_scores_sort_ind[i]]
#             rate = float(one_of_topkFeatures_score / sum_firstorder_scores)
#             firstorder_rate_list.append(rate)
#         elif one_of_topkFeatures.startswith('glcm'):
#             one_of_topkFeatures_score = features_scores[features_scores_sort_ind[i]]
#             rate = float(one_of_topkFeatures_score / sum_glcm_scores)
#             glcm_rate_list.append(rate)
#         else:
#             one_of_topkFeatures_score = features_scores[features_scores_sort_ind[i]]
#             rate = float(one_of_topkFeatures_score / sum_glrlm_scores)
#             glrlm_rate_list.append(rate)
#
#     # 比例列表求和，得出一个总的比例
#     firstorder_rate_list = np.sum(firstorder_rate_list)
#     glcm_rate_list = np.sum(glcm_rate_list)
#     glrlm_rate_list = np.sum(glrlm_rate_list)
#
#     sum_three = firstorder_rate_list + glcm_rate_list + glrlm_rate_list
#
#     # normalize
#     firstorder_rate_normal = float(firstorder_rate_list / sum_three)
#     glcm_rate_normal = float(glcm_rate_list / sum_three)
#     glrlm_rate_normal = float(glrlm_rate_list / sum_three)
#
#     three_features_dict = {'FO': firstorder_rate_normal,
#                            'GLCM': glcm_rate_normal,
#                            'GLRLM': glrlm_rate_normal}
#
#     return three_features_dict

def radar_chart_set(nsel, nclass, feature_selectedrate_dict, save_path,
                    list_feats_selection_args, list_classifers_args):
    fig2, axs2 = plt.subplots(nrows=nsel, ncols=nclass, figsize=(22, 15),
                              subplot_kw=dict(polar=True))
    e = 0
    for row in range(nsel):
        for col in range(nclass):
            features = list(feature_selectedrate_dict[e].keys())
            if len(features) < 3:
                warnings.warn("The number of feature family needs to be ≥ 3")
            counts = list(feature_selectedrate_dict[e].values())
            # calculate angel
            theta = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
            # repeat the last value once to close the radar chart
            counts = np.concatenate((counts, [counts[0]]))
            theta = np.concatenate((theta, [theta[0]]))
            # subplot
            axs2[row, col].fill(theta, counts, 'b', alpha=0.25)
            axs2[row, col].set_title(f'({list_feats_selection_args[row]}, {list_classifers_args[col]})', fontsize=11)
            axs2[row, col].set_xticks(theta[:-1])
            axs2[row, col].set_xticklabels(features)
            # axs2[row, col].set_yticklabels([])

            # for i, count in enumerate(counts[:-1]):
            #     angle = theta[i] + (theta[i + 1] - theta[i]) / 2
            #     radius = count + 0.4
            #
            #     formatted_count = "{:.1%}".format(count)
            #     axs2[row, col].text(angle, radius, formatted_count, ha='center', va='top')
            e += 1
    plt.suptitle('The contribution of different features', fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_path}/Radar_charts.png')
    plt.close()


def findTopKFeatures_rate(fullFeatures, features_scores, featfamilyname_list, topk=10):
    featfamily_num = len(featfamilyname_list)
    overall_featfamily_rate_list = []
    # get feature family
    for i in range(featfamily_num):
        feature_name = featfamilyname_list[i]             # firstorder, glcm, glrlm
        features_scores_sort_ind = np.argsort(features_scores)[::-1]
        feature_indices = [i for i, feature in enumerate(fullFeatures) if feature.startswith(feature_name)]
        feature_scores = [features_scores[i] for i in feature_indices]
        sum_feature_scores = np.sum(feature_scores)
        feature_rate_list = []
        # Iterate through the top k features, get the proportion of each feature's score in its feature family.
        for i in range(topk):
            one_of_topkFeatures = fullFeatures[features_scores_sort_ind[i]]
            if one_of_topkFeatures.startswith(feature_name):
                one_of_topkFeatures_score = features_scores[features_scores_sort_ind[i]]
                rate = float(one_of_topkFeatures_score / sum_feature_scores)
                feature_rate_list.append(rate)
        # Sum the proportion list to calculate an overall proportion
        feature_rate_list = np.sum(feature_rate_list)

        feature_rate_list_value = np.array(feature_rate_list)
        """
        Sum them up. The process before normalization is completed. So far, the results are 17%, 10%, and 50%
        """
        overall_featfamily_rate_list.append(feature_rate_list_value)

    sum_featfamily_num = np.sum(overall_featfamily_rate_list)
    # normalize,for example 17% / (17% + 10% + 50%)
    featfamily_num_features_dict = {}
    for n in range(featfamily_num):
        featfamily_num_features_dict[f'{featfamilyname_list[n]}'] = float(overall_featfamily_rate_list[n] / sum_featfamily_num)
    return featfamily_num_features_dict, featfamily_num



from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
def classifier_measures(y_true, y_pred, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn) 
    error_rate = (fp+fn)/(tp+tn+fp+fn) 
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall) / (precision + recall)
    if np.isnan(y_scores).any():
        ind = np.unique(np.argwhere(np.isnan(y_scores))[:, 0])
        y_scores[ind, 1] = y_pred[ind]
        print('same probabilities got the value of nan! ')
    auc = roc_auc_score(y_true, y_scores[:, 1])

    y_true = pd.DataFrame(y_true)
    y_scores = pd.DataFrame(y_scores)
    y_scores = y_scores.drop(columns=[0])
    fpr, tpr, thresh = roc_curve(y_true, y_scores)    # y_true, y_scores must be one dimensional

    return accuracy, error_rate, sensitivity, specificity, precision, recall, f1, auc, fpr, tpr

# run once to complete a complete five-fold cross-validation for a combination
def run(X, y, selected_features_coorelated_feats, i_featsSelection, j_classifer, list_feats_selection, list_classifers,
        n_features=6, kfold=5, findBestParams=False, dealWithImbalance=True, feature_score_method='addone'):
    
    i = i_featsSelection
    j = j_classifer
    y_test_pred_full = np.zeros(y.shape[0])
    y_test_score_pred_full = np.zeros([y.shape[0], 2])
    selected_features_list0 = [] 
    best_params=None
    features_scores = np.zeros(len(selected_features_coorelated_feats))

    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random.randint(1, 1000))
    # kf = KFold(n_splits=kfold, shuffle=True) ###, random_state=random.randint(1,1000)
    for i_kf, [train_index, test_index] in enumerate(kf.split(X, y)):
    # for i_kf, [train_index, test_index] in enumerate(kf.split(X)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_index_ori, test_index_ori = train_index, test_index
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # X_train, y_train, X_test, y_test = imBalanced_mv_remaining_to_test(X_train, y_train, X_test, y_test )
        X_train = pd.DataFrame(data=X_train, columns=selected_features_coorelated_feats)

        ### feats selection
        X_train, selected_features = list_feats_selection[i](X=X_train, y=y_train, n_features=n_features)
        selected_features_list0.append(selected_features)


        ind_selected_features = findIndexOfFeatures(fullFeatures=selected_features_coorelated_feats, selectedFeatrues=selected_features)
        if feature_score_method == 'addone': 
            ## addone feats selection
            features_scores[ind_selected_features] = features_scores[ind_selected_features] + 1 
        elif feature_score_method == 'weighted': 
            ## weight feats selection. gave the 1st important feat to the highest score (len(n_features)+1) and last important feat to the lowest score (1) 
            features_scores[ind_selected_features] = features_scores[ind_selected_features] + (np.arange(n_features)[::-1]+1) 
        else: 
            raise ValueError(
                f"feature_score_method must be 'addone' or 'weighted'. "
            ) 

        X_test = X_test[:, ind_selected_features]
        # if i_kf==0 and findBestParams:
        #     ## find the Best params for each classifier
        #     ## only find in the 1-st fold in the CV
        #     y_test_pred, y_test_score_pred, best_params = list_classifers[j](X_train, y_train.ravel()).search_params(X_test)
        #     # best_params_list_total[i, j] =  best_params
        # # y_test_pred, y_test_score_pred = list_classifers[j](X_train, y_train, params_dict=best_params).predict(X_test)
        # percent_1 = np.sum(y_train==1) / y_train.shape[0]
        if dealWithImbalance: # and (percent_1 <= 2/5 or percent_1 >= 3/5):
            y_test_pred, y_test_score_pred = imBalanced_MES(estimator=list_classifers[j], X=X_train, y=y_train, X_test=X_test, y_test=y_test, params_dict=best_params)
        else: 
            y_test_pred, y_test_score_pred = list_classifers[j](X_train, y_train, params_dict=best_params).predict(X_test)
        y_test_pred_full[test_index] = y_test_pred
        y_test_score_pred_full[test_index, :] = y_test_score_pred
    return [y_test_pred_full, y_test_score_pred_full, selected_features_list0, features_scores, best_params]

from collections import Counter
def voting_cross_validation_res(results_pack, method, n_samples, y, weight_by_auc=True):
    """
    :param results_pack: list, results of 100 times cross validation
    :param method: str, method for voting (hard_vote, soft_vote, weighted_vote)
    :param n_samples: integer, the number of samples
    :param y: the real label
    :param weight_by_auc: bool, whether weighted for AUC

    Returns:
    ensemble_pred : array
            predicted label
    ensemble_prob : array
            predicted probabilities
    prediction_confidence : array
            predicted confidence
    """
    n_iterations = len(results_pack)    # 100
    all_predictions = np.zeros((n_iterations, n_samples))
    all_probabilities = np.zeros((n_iterations, n_samples))
    all_aucs = np.zeros(n_iterations)

    for i in range(n_iterations):
        y_test_pred_full, y_test_score_pred_full, *_ = results_pack[i]
        _, _, _, _, _, _, _, auc, _, _ = classifier_measures(y, y_test_pred_full, y_test_score_pred_full)
        all_predictions[i] = y_test_pred_full
        # label 1 probabilities
        all_probabilities[i] = y_test_score_pred_full[:, 1] if y_test_score_pred_full.ndim > 1 else y_test_score_pred_full
        all_aucs[i] = auc
    if method == 'hard_vote':
        ensemble_pred = np.zeros(n_samples)
        prediction_confidence = np.zeros(n_samples)
        for i in range(n_samples):
            votes = all_predictions[:, i]
            vote_counts = Counter(votes)
            # the predicted majority label
            ensemble_pred[i] = vote_counts.most_common(1)[0][0]
            # the confidence is the majority label ratio
            prediction_confidence[i] = vote_counts.most_common(1)[0][1] / n_iterations
        # ensemble probabilities are average
        ensemble_prob = np.mean(all_probabilities, axis=0)
    elif method == 'soft_vote':
        # predicted average probabilities decide
        ensemble_prob = np.mean(all_probabilities, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        prob_std = np.std(all_probabilities, axis=0)
        prediction_confidence = 1 - (prob_std / np.max(prob_std))  # normalize [0,1]
    elif method == 'weighted_vote':
        # whether weighted based on AUC
        if weight_by_auc:
            # the higher AUC, the greater weights
            weights = all_aucs / np.sum(all_aucs)
        else:
            # same weights 1 / n_iterations
            weights = np.ones(n_iterations) / n_iterations
        ensemble_prob = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        # weighted variance
        weighted_var = np.average((all_probabilities - ensemble_prob) ** 2, axis=0, weights=weights)
        prediction_confidence = 1 - (weighted_var / np.max(weighted_var))

    return ensemble_pred


from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
def kfold_classifier_featureselection(list_feats_selection, list_classifers, save_dir=None, kfold=10,
                                      df_features_training_uncorr=None, df_total_labels=None, df_patients_id=None,
                                      n_repeats=100, n_features=8, n_workers=12, dealWithImbalance=True,
                                      findBestParams=False, feature_score_method='addone', predict_label_method='weighted_vote'):

    selected_features_coorelated_feats = df_features_training_uncorr.columns.to_list()
    X, y = df_features_training_uncorr.to_numpy(), df_total_labels.to_numpy()

    df_patients_id.reset_index(drop=True, inplace=True)
    df_total_labels.reset_index(drop=True, inplace=True)
    df_features_training_uncorr.reset_index(drop=True, inplace=True)    # violin plot needed

    metrics_feats_selection_classifers = []
    metrics_feats_selection_classifers_fpr = pd.DataFrame()
    metrics_feats_selection_classifers_tpr = pd.DataFrame()
    metrics_feats_selection_classifers_fpr_fill_std = pd.DataFrame()
    metrics_feats_selection_classifers_tpr_fill_std = pd.DataFrame()
    all_the_y_test_pred_full = df_patients_id

    selected_features_list = []
    best_params_list_total = np.empty(shape=[len(list_feats_selection), len(list_classifers)], dtype=object)
    features_scores_total = np.empty(shape=[len(list_feats_selection), len(list_classifers)], dtype=object)
    topkFeatures_total = np.empty(shape=[len(list_feats_selection), len(list_classifers)], dtype=object)
    add = 0
    three_features_dict = {}
    metrics_total = np.empty(shape=[len(list_feats_selection), len(list_classifers)], dtype=object)
    for i in range(len(list_feats_selection)):
        metrics_classifers = []
        metrics_classifers_fpr = pd.DataFrame()
        metrics_classifers_tpr = pd.DataFrame()
        metrics_classifers_fpr_fill_std = pd.DataFrame()
        metrics_classifers_tpr_fill_std = pd.DataFrame()
        auc_list_all = []
        best_params_list = []
        for j in range(len(list_classifers)):
            best_params = None
            metrics = []
            fpr_list = []
            tpr_list = []
            features_scores = np.zeros(len(selected_features_coorelated_feats))
            # kf = KFold(n_splits=kfold, shuffle=True )
            # kf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=100)
            begin_time = time.time()
            if n_workers:
                with Pool(processes=n_workers) as pool:
                    results_pack = pool.starmap(run,
                                                zip(repeat(X, n_repeats),
                                                    repeat(y, n_repeats),
                                                    repeat(selected_features_coorelated_feats, n_repeats),
                                                    repeat(i, n_repeats),
                                                    repeat(j, n_repeats),
                                                    repeat(list_feats_selection, n_repeats),
                                                    repeat(list_classifers, n_repeats),
                                                    repeat(n_features, n_repeats),
                                                    repeat(kfold, n_repeats),
                                                    repeat(findBestParams, n_repeats),
                                                    repeat(dealWithImbalance, n_repeats),
                                                    repeat(feature_score_method, n_repeats),
                                                    ))
            else:
                ## compute features normally
                results_pack = []
                for i_results in range(n_repeats):   #### repeat 10 times
                    [y_test_pred_full, y_test_score_pred_full, selected_features_list0, features_scores0, best_params] = \
                    run(X, y, selected_features_coorelated_feats, i, j, n_features, kfold, findBestParams, dealWithImbalance, feature_score_method)
                    results_pack.append([y_test_pred_full, y_test_score_pred_full, selected_features_list0, features_scores0, best_params])

            features_scores_every = np.empty(shape=[n_repeats, len(selected_features_coorelated_feats)], dtype=int)
            for i_pack in range(len(results_pack)):
                [y_test_pred_full, y_test_score_pred_full, selected_features_list0, features_scores0, best_params] = results_pack[i_pack]
                features_scores = features_scores + features_scores0
                features_scores_every[i_pack] = features_scores0
                [accuracy, error_rate, sensitivity, specificity, precision, recall, f1, auc, fpr, tpr] = classifier_measures(y, y_test_pred_full, y_test_score_pred_full)
                metrics.append([accuracy, error_rate, sensitivity, specificity, precision, recall, f1, auc])
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                best_params_list.append(best_params)
                # retain the prediction results corresponding to the highest AUC in 100 times of five-fold cross-validation for each combination.

            voting_y_test_pred_full = voting_cross_validation_res(results_pack, predict_label_method,
                                                                  len(df_patients_id), y, True)
            feats_selection_name = str(list_feats_selection[i]).split(' ')[1]
            classifiers_name = str(list_classifers[j]).split('.')[-1][:-2]

            # return the prediction results corresponding to the voting under all combinations
            voting_y_test_pred_full = pd.DataFrame(voting_y_test_pred_full)
            voting_y_test_pred_full = voting_y_test_pred_full.rename(columns=lambda x: f'{feats_selection_name}_{classifiers_name}')
            all_the_y_test_pred_full = pd.concat([all_the_y_test_pred_full, voting_y_test_pred_full], axis=1)

            best_params_list_total[i, j] = best_params_list
            print('finished: ', i, list_feats_selection[i], j, list_classifers[j])
            end_time = time.time()
            print("time: ", end_time-begin_time)

            # np.set_printoptions(precision=4)  ## set 4 decimal places
            metrics_np = np.array(metrics)
            fpr_list = pd.DataFrame(fpr_list)
            tpr_list = pd.DataFrame(tpr_list)
            # mean in col
            metrics_mean = np.mean(metrics_np, axis=0)
            metrics_mean_fpr = fpr_list.mean(skipna=True, axis=0)
            metrics_mean_tpr = tpr_list.mean(skipna=True, axis=0)

            # std in col
            metrics_std = np.std(metrics_np, axis=0)
            metrics_std_fpr = fpr_list.std(skipna=True, axis=0)
            metrics_std_tpr = tpr_list.std(skipna=True, axis=0)

            sort_indices = np.argsort(metrics_mean_fpr)
            metrics_mean_fpr_sorted = metrics_mean_fpr[sort_indices]
            metrics_mean_tpr_sorted = metrics_mean_tpr[sort_indices]
            metrics_std_tpr_sorted = metrics_std_tpr[sort_indices]

            # AUC
            AUROC_value = AUROC(metrics_mean_fpr_sorted, metrics_mean_tpr_sorted)
            # Plot ROC Curve
            # plt.figure(figsize=(8, 8))
            # plt.plot(metrics_mean_fpr_sorted, metrics_mean_tpr_sorted, color='darkorange', lw=2, label=f'ROC (AUC = {AUROC_value:.4f})')   # label=f'ROC = {auc:.4f}'
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.fill_between(metrics_mean_fpr_sorted, metrics_mean_tpr_sorted - metrics_std_tpr_sorted, metrics_mean_tpr_sorted + metrics_std_tpr_sorted,
            #                  color='skyblue', alpha=0.5, label='Std')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title(f'ROC Curve;Selection:{feats_selection_name};Classifiers:{classifiers_name}')
            # plt.legend(loc='lower right')
            # # plt.show()
            # folder_time1 = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
            # plt.savefig(
            #     f'./visual_select_times/{feats_selection_name}_{classifiers_name}_{folder_time1}_ROC_Curve.png')
            # plt.close()

            matrics_str = [r"{:.4g} / {:.4g}".format(metrics_mean[metr_i], metrics_std[metr_i]) for metr_i in range(len(metrics_mean))]
            metrics_classifers.append(matrics_str)

            metrics_classifers_fpr = pd.concat([metrics_classifers_fpr, metrics_mean_fpr], axis=1)
            metrics_classifers_tpr = pd.concat([metrics_classifers_tpr, metrics_mean_tpr], axis=1)
            metrics_classifers_fpr_fill_std = pd.concat([metrics_classifers_fpr_fill_std, metrics_std_fpr], axis=1)
            metrics_classifers_tpr_fill_std = pd.concat([metrics_classifers_tpr_fill_std, metrics_std_tpr], axis=1)

            # metrics_classifers.append(metrics_mean)
            features_scores_total[i, j] = features_scores
            topkFeatures_total[i, j] = findTopKFeatures(fullFeatures=selected_features_coorelated_feats, features_scores=features_scores, topk=n_features)
            metrics_total[i, j] = metrics_np

            featfamilyname = set()
            for col in selected_features_coorelated_feats:
                prefix = col.split('_')[0]
                featfamilyname.add(prefix)
            featfamilyname_list = list(featfamilyname)
            three_features_dict[add], featfamily_num = findTopKFeatures_rate(fullFeatures=selected_features_coorelated_feats,
                                                                             features_scores=features_scores,
                                                                             featfamilyname_list=featfamilyname_list,
                                                                             topk=n_features)
            add += 1

            # top 12 features
            top2k = int(2*n_features)
            top12_features_indices = np.argsort(-features_scores)[:top2k]
            top12_features = [selected_features_coorelated_feats[i] for i in top12_features_indices]

            top12_features_indices_every = [selected_features_coorelated_feats.index(feature) for feature in
                                            top12_features]
            features_scores_every_drop = features_scores_every[:, top12_features_indices_every]

            features_scores_every_drop_df = pd.DataFrame(features_scores_every_drop, columns=top12_features)

            top2k_half = int(n_features)
            top12_features_indices_half = np.argsort(-features_scores)[:top2k_half]
            top12_features_half = [selected_features_coorelated_feats[i] for i in top12_features_indices_half]

            # Violin plot of top features in every combination
            row_num, row_col = calculate_layout(n_features)
            violin_plot(row_num, row_col, top12_features_half, df_features_training_uncorr,
                        df_total_labels, feats_selection_name, classifiers_name,
                        save_path=save_dir)

            # top 12 feature
            top_12_scores = features_scores[top12_features_indices]
            total_selections = top_12_scores.sum()
            top_12_scores_percentage = (top_12_scores / total_selections) * 100
            # Horizontal bar plot of top features in every combination
            horizontal_bar_chart(top12_features, top_12_scores_percentage,
                                 feats_selection_name, classifiers_name,
                                 save_path=save_dir)

        metrics_feats_selection_classifers.append(metrics_classifers)
        metrics_feats_selection_classifers_fpr = pd.concat([metrics_feats_selection_classifers_fpr, metrics_classifers_fpr], axis=1)
        metrics_feats_selection_classifers_tpr = pd.concat([metrics_feats_selection_classifers_tpr, metrics_classifers_tpr], axis=1)
        metrics_feats_selection_classifers_fpr_fill_std = pd.concat([metrics_feats_selection_classifers_fpr_fill_std, metrics_classifers_fpr_fill_std], axis=1)
        metrics_feats_selection_classifers_tpr_fill_std = pd.concat([metrics_feats_selection_classifers_tpr_fill_std, metrics_classifers_tpr_fill_std], axis=1)

    metrics_feats_selection_classifers = np.array(metrics_feats_selection_classifers)

    return (metrics_feats_selection_classifers, metrics_total, features_scores_total, topkFeatures_total,
            best_params_list_total, metrics_feats_selection_classifers_fpr, metrics_feats_selection_classifers_tpr,
            metrics_feats_selection_classifers_fpr_fill_std, metrics_feats_selection_classifers_tpr_fill_std,
            three_features_dict, all_the_y_test_pred_full)


def move_files(source_path, destination_path):
    files = glob.glob(os.path.join(source_path, '*'))
    for file in files:
        destination_file_path = os.path.join(destination_path, os.path.basename(file))

        shutil.move(file, destination_file_path)


def cross_validation(dataset_feature_matrix: Path, survival_info_dir: Path, save_dir: Path, top_feature_num=6, k_fold=5,
                     feature_score_method="addone", var_thresh=0, corr_threshold=0.9, repeats_num=100,
                     list_feats_selection_args=None, list_classifers_args=None, n_workers=10):
    """
    Perform multi-fold cross-validation based on dataset feature matrix

    :param dataset_feature_matrix: str, the directory of dataset feature_matrix (csv)
    :param survival_info_dir: str, the directory of survival information, the first column is "wsi_id", the second
    column is "survival_time (month)", and the third column is event_status (1 is death, 0 is survival)
    :param save_dir: str, the directory of saving results
    :param top_feature_num: integer, the number of top features
    :param k_fold: integer, the number of folds
    :param feature_score_method: str, the method of calculating feature score. "addone" or "weighted"
    :param var_thresh: float, default 0, remove the redundant features with variance
    :param corr_threshold: float, default 0.9, remove the redundant features with correlation
    :param repeats_num: integer, times to repeat the experiment
    :param list_feats_selection_args: list, 10 feature selection methods,['Lasso', 'XGBoost', 'RandomForest',
    'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest', 'ranksums', 'mutualInfo']
    :param list_classifers_args: list, 11 classifiers, ['QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh',
    'LinearSVC', 'MLP', 'GaussianNB', 'SGD', 'SVC_rbf', 'AdaBoost']
    :param n_workers: the number of processes

    :return: list, the best combination, [feature selection method, classifier] (e.g., ['mrmr', 'QDA'])
    """
    list_feats_selection_Dict = {'Univariate': UnivariateFeatureSelection,
                                 'mutualInfo': mutual_info_selection,
                                 'mrmr': mrmr_selection,
                                 'ttest': ttest_selection,
                                 'ranksums': ranksums_selection,
                                 'RFE': rfe_selection,
                                 'RandomForest': random_forest_slectiion,
                                 'Elastic-Net': elastic_net_slectiion,
                                 'XGBoost': XGBoost_slectiion,
                                 'Lasso': lasso_feature_selection}
    list_classifers_Dict = {'QDA': QDA_,
                            'LDA': LDA_,
                            'RandomForest': RandomForestC_,
                            'DecisionTree': DecisionTree_,
                            'KNeigh': KNeighbors_,
                            'LinearSVC': LinearSVC_,
                            'MLP': MLPClassifier_,
                            'GaussianNB': GaussianNB_,
                            'SGD': SGDClassifier_,
                            'SVC_rbf': SVC_rbf_,
                            'AdaBoost': AdaBoostClassifier_}
    list_feats_selection = [list_feats_selection_Dict[list_feats_selection_args[i]] for i in range(len(list_feats_selection_args))]
    list_classifers = [list_classifers_Dict[list_classifers_args[i]] for i in range(len(list_classifers_args))]
    save_dir.mkdir(parents=True, exist_ok=True)

    df_features_labels = pd.read_csv(dataset_feature_matrix)
    if len(df_features_labels) < 16:
        warnings.warn("Please ensure an adequate number of samples (＞16) and balanced categories as much as possible "
                      "to prevent classifier errors.")
    df_labels = df_features_labels[['label']]
    df_wsi_ids = df_features_labels[['wsi_id']]
    df_features = df_features_labels.drop(columns=['wsi_id', 'label'])
    #  remove the constant varibles
    df_total, selected_featsName_from_var_thresh = varianceThreshold_rm(X=df_features, var_thresh=var_thresh)
    # remove redundant variables
    df_features_training_uncorr, dropped_featsName = remove_correlated_features(df_total, corr_threshold=corr_threshold)
    # z-scores
    scaler = StandardScaler()
    df_features_training_uncorr = pd.DataFrame(data=scaler.fit_transform(df_features_training_uncorr),
                                               columns=df_features_training_uncorr.columns)
    #### the 1st time select the uncorrelation feature
    df_features_training_uncorr_label_ptID = pd.concat([df_features_training_uncorr, df_labels, df_wsi_ids],
                                                       axis=1).reset_index(drop=True)
    zeros_num = df_labels[df_labels['label'] == 0]
    print('0: ', zeros_num.shape[0], '1: ', df_labels.shape[0] - zeros_num.shape[0])

    #### the 2nd time select the uncorrelation feature but not just for selection
    # matrix_M_patch = f'/media/linjianwei/data_16T/output/Paper_Pypathomics(example)/M_matrix_TCGA_178_with_label.csv'
    (metrics_feats_selection_classifers, metrics_total, features_scores_total, topkFeatures_total,
     best_params_list_total,
     metrics_feats_selection_classifers_fpr, metrics_feats_selection_classifers_tpr,
     metrics_feats_selection_classifers_fpr_fill_std, metrics_feats_selection_classifers_tpr_fill_std,
     three_features_dict, all_the_y_test_pred_full) \
        = kfold_classifier_featureselection(list_feats_selection, list_classifers, save_dir, k_fold,
                                            df_features_training_uncorr, df_labels, df_wsi_ids,
                                            n_repeats=repeats_num, n_features=top_feature_num,
                                            n_workers=n_workers,
                                            feature_score_method=feature_score_method)  ##addone
    #### just get 8 metricses including acc err sen spe pre rec f1 auc and save them.
    df_acc_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 0],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_err_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 1],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_sen_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 2],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_spe_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 3],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_pre_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 4],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_rec_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 5],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_f1_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 6],
                               index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                               columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_auc_table = pd.DataFrame(data=metrics_feats_selection_classifers[:, :, 7],
                                index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                columns=[str(cla_nm) for cla_nm in list_classifers_args])

    df_auc_table_copy = df_auc_table.copy()
    df_auc_table_copy.rename_axis('selection', axis='index', inplace=True)
    # df_auc_table_copy.set_index('selection', inplace=True)
    df_auc_table_copy_std = df_auc_table_copy.copy()
    for col in df_auc_table_copy_std.columns:
        df_auc_table_copy_std[col] = df_auc_table_copy_std[col].apply(lambda x: float(x.split('/')[1]))
    for col in df_auc_table_copy.columns:
        df_auc_table_copy[col] = df_auc_table_copy[col].apply(lambda x: float(x.split('/')[0]))

    # The largest value in all
    max = df_auc_table_copy.max().max()
    max_loc = np.where(df_auc_table_copy == max)
    if len(max_loc[0]) > 1:  # not only one max
        print('multiple max values')
        # find the lowest std from these max values
        min_std = [df_auc_table_copy_std.iloc[max_loc[0][i], max_loc[1][i]] for i in range(len(max_loc[0]))]
        min_std = pd.DataFrame(min_std)
        min_std = min_std.min().min()
        # if there is multi min, choose the first one
        min_std_loc = np.where(df_auc_table_copy_std == min_std)
        max_loc = (min_std_loc[0][0], min_std_loc[1][0])

    # get the column name and index name
    max_row = df_auc_table_copy.index[max_loc[0]]
    max_col = df_auc_table_copy.columns[max_loc[1]]

    if isinstance(max_col, str) is False:
        max_col = max_col.values[0]
    if isinstance(max_row, str) is False:
        max_row = max_row.values[0]

    # max_col, max_row
    # print("The best model", max_row, max_col)
    best_combination = [max_row, max_col]
    df_auc_table_copy_reset = df_auc_table_copy.to_numpy()
    df_auc_table_copy_reset = pd.DataFrame(df_auc_table_copy_reset)
    max_row_ind = df_auc_table_copy_reset.index[max_loc[0]]
    max_col_ind = df_auc_table_copy_reset.columns[max_loc[1]]


    # 保存各个组合100次五折交叉验证中最优auc对应的预测结果和预测概率，并添加生存时间和最终状态，用于后续画KM Curve
    # 保存预测结果                 useful_data_TCGA是生存资料
    if survival_info_dir.suffix == '.csv':
        survival_info = pd.read_csv(survival_info_dir)
    elif survival_info_dir in ['.xlsx', '.xls']:
        survival_info = pd.read_excel(survival_info_dir)
    else:
        raise ValueError(f"Supported input formats: .csv, .xlsx, .xls")
    required_columns = ['wsi_id', 'survival_time', 'event_status']
    for col in required_columns:
        if col not in survival_info.columns.to_list():
            raise ValueError(f'Please ensure that the first column is "wsi_id", '
                             f'the second column is "survival_time (month)", and the third column is '
                             f'"event_status (1 is death, 0 is survival)"')

    all_the_y_test_pred_full_with_time = pd.merge(all_the_y_test_pred_full, survival_info,
                                                       on='wsi_id', how='inner')
    all_the_y_test_pred_full_with_time.to_csv(f'{save_dir}/all_the_y_test_pred_full_with_time.csv',
                                                   index=False)

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    results_to_save_path = str(save_dir)
    if not os.path.exists(results_to_save_path):
        os.mkdir(results_to_save_path)

    # radar chart,5*11
    radar_chart_set(len(list_feats_selection_args), len(list_classifers_args), three_features_dict,
                    results_to_save_path, list_feats_selection_args, list_classifers_args)
    # ROC，5*11
    ROC_Chart_set(len(list_feats_selection_args), len(list_classifers_args), metrics_feats_selection_classifers_fpr,
                  metrics_feats_selection_classifers_tpr, metrics_feats_selection_classifers_fpr_fill_std,
                  metrics_feats_selection_classifers_tpr_fill_std, df_auc_table, results_to_save_path,
                  list_feats_selection_args, list_classifers_args, max_row_ind, max_col_ind)

    df_auc_table.to_csv(results_to_save_path + '/auc_table.csv')
    df_acc_table.to_csv(results_to_save_path + '/acc_table.csv')
    df_err_table.to_csv(results_to_save_path + '/err_table.csv')
    df_sen_table.to_csv(results_to_save_path + '/sen_table.csv')
    df_spe_table.to_csv(results_to_save_path + '/spe_table.csv')
    df_pre_table.to_csv(results_to_save_path + '/pre_table.csv')
    df_rec_table.to_csv(results_to_save_path + '/rec_table.csv')
    df_f1_table.to_csv(results_to_save_path + '/f1_table.csv')

    np.save(results_to_save_path + '/metrics_feats_selection_classifers.npy',
            metrics_feats_selection_classifers)
    np.save(results_to_save_path + '/metrics_total.npy', metrics_total)

    df_features_scores_table = pd.DataFrame(data=features_scores_total,
                                            index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                            columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_features_scores_table.to_csv(results_to_save_path + '/features_scores.csv')

    df_topkFeatures_table = pd.DataFrame(data=topkFeatures_total,
                                         index=[str(feats_nm) for feats_nm in list_feats_selection_args],
                                         columns=[str(cla_nm) for cla_nm in list_classifers_args])
    df_topkFeatures_table.to_csv(results_to_save_path + '/topk_features.csv')

    print('finished. the results have been saved in the folder of {}'.format(results_to_save_path))

    return best_combination



def main():
    parser = argparse.ArgumentParser(description='Multi-fold cross validation')
    parser.add_argument('--dataset_feature_matrix', default='../../example_folder/aggregation/dataset_feature_matrix.csv', type=Path, required=True, help='Dataset_feature_matrix.csv')
    parser.add_argument('--survival_info_dir', default='../../example_folder/survival_info.csv', type=Path, required=True, help='Survival_info.csv')
    parser.add_argument('--cross_save_dir', default='../../example_folder/cross_validation', type=Path, required=True)
    args = parser.parse_args()
    best_combination = cross_validation(args.dataset_feature_matrix, args.survival_info_dir, args.cross_save_dir, top_feature_num=6, k_fold=5,
                                        feature_score_method="addone", var_thresh=0, corr_threshold=0.9, repeats_num=100,
                                        list_feats_selection_args=['Lasso', 'XGBoost', 'RandomForest', 'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest', 'ranksums', 'mutualInfo'],
                                        list_classifers_args=['QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh', 'LinearSVC', 'MLP', 'GaussianNB', 'SGD', 'SVC_rbf', 'AdaBoost'],
                                        n_workers=25)
    print("The best combination is {}".format(best_combination))
    # 'Lasso', 'XGBoost', 'RandomForest', 'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest', 'ranksums', 'mutualInfo'
    # 'QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh', 'LinearSVC', 'MLP', 'GaussianNB', 'SGD', 'SVC_rbf', 'AdaBoost'


if __name__ == '__main__':
    main()







    
