import warnings

import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# Import your custom classifiers and feature selection methods
# These should be available in your environment or adjust the imports accordingly
from Bclassifiers import QDA_, LDA_, RandomForestC_, SGDClassifier_, DecisionTree_, \
                        KNeighbors_, LinearSVC_, SVC_rbf_, GaussianProcess_, \
                            MLPClassifier_, AdaBoostClassifier_, GaussianNB_, Ridge_
from Bfeatures_selection import (UnivariateFeatureSelection, mutual_info_selection, mrmr_selection, \
                            ttest_selection, ranksums_selection, varianceThreshold_rm, rfe_selection, chi_select,
                            elastic_net_slectiion, random_forest_slectiion, XGBoost_slectiion, lasso_feature_selection)

class SurvivalAnalysisPlotter:
    def __init__(self, cross_validation: Path, list_feats_selection=None, list_classifiers=None, n_workers=4):
        """
        Initializes the SurvivalAnalysisPlotter with data and parameters.

        :param cross_validation: Path to the cross validation results .
        :param list_feats_selection: List of feature selection methods.
        :param list_classifiers: List of classifier methods.
        :param n_workers: Number of workers for parallel processing.
        """
        self.cross_validation = cross_validation
        self.data_file = self.cross_validation / "all_the_y_test_pred_full_with_time.csv"
        self.n_workers = n_workers
        self.list_feats_selection_Dict = {
            'Univariate': UnivariateFeatureSelection,
            'mutualInfo': mutual_info_selection,
            'mrmr': mrmr_selection,
            'ttest': ttest_selection,
            'ranksums': ranksums_selection,
            'RFE': rfe_selection,
            'RandomForest': random_forest_slectiion,
            'Elastic-Net': elastic_net_slectiion,
            'XGBoost': XGBoost_slectiion,
            'Lasso': lasso_feature_selection
        }
        self.list_classifiers_Dict = {
            'QDA': QDA_,
            'LDA': LDA_,
            'RandomForest': RandomForestC_,
            'DecisionTree': DecisionTree_,
            'KNeigh': KNeighbors_,
            'LinearSVC': LinearSVC_,
            'MLP': MLPClassifier_,
            'GaussianNB': GaussianNB_,
            'SGD': SGDClassifier_,
            'SVC_rbf': SVC_rbf_,
            'AdaBoost': AdaBoostClassifier_,
            'Ridge': Ridge_,
            'GaussianProcess': GaussianProcess_
        }
        self.list_feats_selection = list_feats_selection
        self.list_classifiers = list_classifiers
        self.data = None
        self.best_combination = None
        self.num_select = None
        self.num_classify = None
        self.processed_feats_selection = []
        self.processed_classifiers = []
        self.load_data()
        self.process_methods()
        self.get_best_combination()

    def load_data(self):
        """Loads data from the CSV file and preprocesses it."""
        self.data = pd.read_csv(self.data_file)
        # self.data = self.data.drop(columns=['patient_id_match', 'ID', 'patient_id'])

    def process_methods(self):
        """Processes feature selection and classifier methods based on provided lists."""
        self.processed_feats_selection = [
            self.list_feats_selection_Dict[method] for method in self.list_feats_selection
        ]
        self.processed_classifiers = [
            self.list_classifiers_Dict[method] for method in self.list_classifiers
        ]

    def get_best_combination(self):
        auc_tabel = self.cross_validation / "auc_table.csv"
        auc_tabel = pd.read_csv(auc_tabel)
        self.num_select = auc_tabel.shape[0]
        self.num_classify = auc_tabel.shape[1] - 1

        # handle auc
        auc_tabel.rename(columns={'Unnamed: 0': 'selection'}, inplace=True)
        auc_tabel.set_index('selection', inplace=True)
        auc_tabel_std = auc_tabel.copy()
        for col in auc_tabel_std.columns:
            auc_tabel_std[col] = auc_tabel_std[col].apply(lambda x: float(x.split('/')[1]))
        for col in auc_tabel.columns:
            auc_tabel[col] = auc_tabel[col].apply(lambda x: float(x.split('/')[0]))

        # the largest auc
        max = auc_tabel.max().max()
        max_loc = np.where(auc_tabel == max)
        if len(max_loc[0]) > 1:  # not only one max
            # find the lowest std from these max values
            min_std = [auc_tabel_std.iloc[max_loc[0][i], max_loc[1][i]] for i in range(len(max_loc[0]))]
            min_std = pd.DataFrame(min_std)
            min_std = min_std.min().min()
            # min_std = metric_std.iloc[max_loc[0], max_loc[1]].min()
            # min_std = min_std.min()
            # if there is multi min, choose the first one
            min_std_loc = np.where(auc_tabel_std == min_std)
            max_loc = (min_std_loc[0][0], min_std_loc[1][0])

        # get the column name and index name
        max_row = auc_tabel.index[max_loc[0]]
        max_col = auc_tabel.columns[max_loc[1]]
        if isinstance(max_col, str) is False:
            max_col = max_col.values[0]
        if isinstance(max_row, str) is False:
            max_row = max_row.values[0]

        self.best_combination = [max_row, max_col]

    def km_curve_set(self):
        """
        Plots KM curves for each combination of feature selection and classifier methods.
        """
        fig, ax = plt.subplots(nrows=self.num_select, ncols=self.num_classify, figsize=(34, 62))
        column_names = list(self.data.columns[1:-3])
        x = 0
        for nrow in range(self.num_select):
            for ncol in range(self.num_classify):
                feats_selection_name = str(self.processed_feats_selection[nrow]).split(' ')[1]
                classifiers_name = str(self.processed_classifiers[ncol]).split('.')[-1][:-2]
                feature = column_names[x]
                subset_data = self.data[[feature, 'survival_time', 'event_status']]
                cph = CoxPHFitter()
                cph.fit(subset_data, duration_col='survival_time', event_col='event_status')
                hr = cph.summary['exp(coef)'][0]
                ci_lower = cph.summary['exp(coef) lower 95%'][0]
                ci_higher = cph.summary['exp(coef) upper 95%'][0]
                # Replace numerical labels with risk categories
                subset_data[feature] = subset_data[feature].replace({0: 'Long term', 1: 'Short term'})
                for i in subset_data[feature].unique():
                    kmf = KaplanMeierFitter()
                    df_tmp = subset_data.loc[subset_data[feature] == i]
                    kmf.fit(df_tmp['survival_time'],
                            event_observed=df_tmp['event_status'],
                            label=i)
                    line_color = '#2878B5' if i == 'Long term' else '#c82423'
                    kmf.plot_survival_function(ci_show=False, ax=ax[nrow, ncol], color=line_color, show_censors=True)
                # Log-rank test
                p_value = multivariate_logrank_test(
                    event_durations=subset_data['survival_time'],
                    groups=subset_data[feature],
                    event_observed=subset_data['event_status']
                ).p_value
                x += 1
                p_value_text = 'p-value < 0.001' if p_value < 0.001 else f'p-value = {p_value:.4F}'
                ax[nrow, ncol].set_title(f"{feature}", fontsize=8)
                ax[nrow, ncol].text(0.05, 0.08, p_value_text, fontsize=8, transform=ax[nrow][ncol].transAxes)
                ax[nrow, ncol].text(0.05, 0.04, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})",
                                    fontsize=8, transform=ax[nrow][ncol].transAxes)
                ax[nrow, ncol].set_xlabel("Months", fontsize=8)
                ax[nrow, ncol].set_ylabel("Survival Probabilities", fontsize=8)
                ax[nrow, ncol].tick_params(axis='x', labelsize=8)
                ax[nrow, ncol].tick_params(axis='y', labelsize=8)
                ax[nrow, ncol].legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def km_curve_every(self, save_dir: Path) -> None:
        """
        Plots and saves individual KM curves for each combination of feature selection and classifier methods.

        :param save_dir: Directory to save the plots.
        """
        save_dir = save_dir / "km_curve_every"
        os.makedirs(save_dir, exist_ok=True)
        column_names = list(self.data.columns[1:-3])
        x = 0
        for nrow in range(self.num_select):
            for ncol in range(self.num_classify):
                feats_selection_name = str(self.processed_feats_selection[nrow]).split(' ')[1]
                classifiers_name = str(self.processed_classifiers[ncol]).split('.')[-1][:-2]
                feature = column_names[x]
                subset_data = self.data[[feature, 'survival_time', 'event_status']]
                cph = CoxPHFitter()
                cph.fit(subset_data, duration_col='survival_time', event_col='event_status')
                hr = cph.summary['exp(coef)'][0]
                ci_lower = cph.summary['exp(coef) lower 95%'][0]
                ci_higher = cph.summary['exp(coef) upper 95%'][0]
                fig, ax = plt.subplots()
                subset_data[feature] = subset_data[feature].replace({0: 'Long term', 1: 'Short term'})
                n = 0
                for i in subset_data[feature].unique():
                    kmf = KaplanMeierFitter()
                    df_tmp = subset_data.loc[subset_data[feature] == i]
                    kmf.fit(df_tmp['survival_time'], event_observed=df_tmp['event_status'], label=i)
                    line_color = '#2878B5' if i == 'Long term' else '#c82423'
                    kmf.plot_survival_function(ax=ax, ci_show=False, color=line_color, show_censors=True)
                    add_at_risk_counts(kmf, ax=ax, labels=f'{i}', rows_to_show=['At risk', 'Censored'], ypos=-0.26 - n)
                    n += 0.65
                p_value = multivariate_logrank_test(
                    event_durations=subset_data['survival_time'],
                    groups=subset_data[feature],
                    event_observed=subset_data['event_status']).p_value
                x += 1
                p_value_text = 'p-value < 0.001' if p_value < 0.001 else f'p-value = {p_value:.4F}'
                ax.set_title(f"{feature}", fontsize=12)
                ax.text(0.22, 0.47, p_value_text, fontsize=8, transform=plt.gcf().transFigure)
                ax.text(0.22, 0.44, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})",
                        fontsize=8, transform=plt.gcf().transFigure)
                ax.set_xlabel("Months", fontsize=12)
                ax.set_ylabel("Survival Probabilities", fontsize=12)
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                plt.tight_layout()
                plot_filename = f"{feats_selection_name}_{classifiers_name}.png"
                plt.savefig(os.path.join(save_dir, plot_filename))
                plt.close(fig)

    def km_curve_best_auc(self, save_path: Path) -> None:
        """
        Plots the KM curve for the best AUC combination(self.best_combination[0], self.best_combination[1]) of
        feature selection and classifier.

        :param save_path: Path to save the KM curve plot.
        """
        save_path = save_path / "km_curve_best_auc"
        os.makedirs(save_path, exist_ok=True)
        best_col = self.data.filter(like=self.best_combination[0]).filter(like=self.best_combination[1])
        if best_col.empty:
            warnings.warn(f"No matching data found for feature selection '{self.best_combination[0]}' and classifier '{self.best_combination[1]}'.")
            return
        best_col_name = best_col.columns.tolist()[0]
        subset_data = self.data[['survival_time', 'event_status']]
        data = pd.concat([best_col, subset_data], axis=1)
        cph = CoxPHFitter()
        cph.fit(data, duration_col='survival_time', event_col='event_status')
        hr = cph.summary['exp(coef)'][0]
        ci_lower = cph.summary['exp(coef) lower 95%'][0]
        ci_higher = cph.summary['exp(coef) upper 95%'][0]
        best_col = best_col.replace({0: 'Long term', 1: 'Short term'})
        best_col = best_col.squeeze()
        fig, ax = plt.subplots()
        n = 0
        for pre_type in best_col.unique():
            sub_data = self.data[best_col == pre_type]
            kmf = KaplanMeierFitter().fit(sub_data['survival_time'], event_observed=sub_data['event_status'], label=f'{pre_type}')
            line_color = '#2878B5' if pre_type == 'Long term' else '#c82423'
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=line_color)
            add_at_risk_counts(kmf, ax=ax, labels=f'{pre_type}', rows_to_show=['At risk', 'Censored'], ypos=-0.26 - n)
            n += 0.65
        p_value = multivariate_logrank_test(
            event_durations=self.data['survival_time'],
            groups=best_col,
            event_observed=self.data['event_status']
        ).p_value
        p_value_text = 'p-value < 0.001' if p_value < 0.001 else f'p-value = {p_value:.4F}'
        ax.text(0.21, 0.45, p_value_text, fontsize=8, transform=plt.gcf().transFigure)
        ax.text(0.21, 0.4, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})", fontsize=8, transform=plt.gcf().transFigure)
        ax.legend()
        ax.set_title(best_col_name)
        ax.set_xlabel('Months')
        ax.set_ylabel('Survival Probabilities')
        plt.tight_layout()
        plot_filename = f"KM_Curve_{best_col_name}.png"
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close(fig)
        print(f"KM curve saved to {os.path.join(save_path, plot_filename)}")

def main():
    parser = argparse.ArgumentParser(description="Survival analysis")
    parser.add_argument('--cross_validation', default="../../example_folder/cross_validation", type=Path, required=True, help="Path to cross-validation result")
    parser.add_argument('--list_feats_selection', nargs='+', required=False,
                        default=['Lasso', 'XGBoost', 'RandomForest', 'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest','ranksums', 'mutualInfo'],
                        help="List of feature selection methods.")
    parser.add_argument('--list_classifiers', nargs='+', required=False,
                        default=['QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh', 'LinearSVC', 'MLP', 'GaussianNB','SGD', 'SVC_rbf', 'AdaBoost'],
                        help="List of classifier methods.")
    parser.add_argument('--n_workers', type=int, default=25, required=True, help="Number of workers for parallel processing.")
    parser.add_argument('--survival_save_dir', type=Path, default="../../example_folder/KM_curve_survival_analysis", required=True, help="Path of save directory")
    args = parser.parse_args()
    plotter = SurvivalAnalysisPlotter(cross_validation=args.cross_validation, list_feats_selection=args.list_feats_selection,
                                      list_classifiers=args.list_classifiers,  n_workers=args.n_workers)
    plotter.km_curve_every(save_dir=args.survival_save_dir)
    plotter.km_curve_best_auc(save_path=args.survival_save_dir)


if __name__ == '__main__':
    main()