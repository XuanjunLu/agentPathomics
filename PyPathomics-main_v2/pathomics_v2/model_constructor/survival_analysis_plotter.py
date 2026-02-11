import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# Import your custom classifiers and feature selection methods
# These should be available in your environment or adjust the imports accordingly
from Bclassifiers import (
    QDA_, LDA_, RandomForestC_, SGDClassifier_, DecisionTree_,
    KNeighbors_, LinearSVC_, SVC_rbf_, GaussianProcess_,
    MLPClassifier_, AdaBoostClassifier_, GaussianNB_, Ridge_
)
from Bfeatures_selection import (
    UnivariateFeatureSelection, mutual_info_selection, mrmr_selection,
    ttest_selection, ranksums_selection, varianceThreshold_rm,
    rfe_selection, chi_select
)

class SurvivalAnalysisPlotter:
    def __init__(self, data_file, list_feats_selection=None, list_classifiers=None, n_workers=4):
        """
        Initializes the SurvivalAnalysisPlotter with data and parameters.

        :param data_file: Path to the CSV file containing prediction results and survival data.
        :param list_feats_selection: List of feature selection methods.
        :param list_classifiers: List of classifier methods.
        :param n_workers: Number of workers for parallel processing.
        """
        self.data_file = data_file
        self.n_workers = n_workers
        self.list_feats_selection_Dict = {
            'Univariate': UnivariateFeatureSelection,
            'mutualInfo': mutual_info_selection,
            'mrmr': mrmr_selection,
            'ttest': ttest_selection,
            'ranksums': ranksums_selection,
            'rfe': rfe_selection,
            'chi': chi_select
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
        self.list_feats_selection = list_feats_selection or ['Univariate', 'mrmr', 'ttest', 'ranksums']
        self.list_classifiers = list_classifiers or ['QDA', 'LDA', 'RandomForest', 'LinearSVC', 'MLP', 'GaussianNB', 'AdaBoost']
        self.data = None
        self.processed_feats_selection = []
        self.processed_classifiers = []
        self.load_data()
        self.process_methods()

    def load_data(self):
        """Loads data from the CSV file and preprocesses it."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_file)
        self.data = self.data.drop(columns=['patient_id_match', 'ID', 'patient_id'])
        print("Data loaded successfully.")

    def process_methods(self):
        """Processes feature selection and classifier methods based on provided lists."""
        self.processed_feats_selection = [
            self.list_feats_selection_Dict[method] for method in self.list_feats_selection
        ]
        self.processed_classifiers = [
            self.list_classifiers_Dict[method] for method in self.list_classifiers
        ]

    def km_curve_set(self, num_row, num_col):
        """
        Plots KM curves for each combination of feature selection and classifier methods.

        :param num_row: Number of rows in the subplot grid.
        :param num_col: Number of columns in the subplot grid.
        """
        fig, ax = plt.subplots(nrows=num_row, ncols=num_col, figsize=(34, 62))
        column_names = list(self.data.columns[0:-2])
        x = 0
        for nrow in range(num_row):
            for ncol in range(num_col):
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
                self.data[feature] = self.data[feature].replace({0: 'Low-risk', 1: 'High-risk'})
                for i in self.data[feature].unique():
                    kmf = KaplanMeierFitter()
                    df_tmp = self.data.loc[self.data[feature] == i]
                    kmf.fit(df_tmp['survival_time'],
                            event_observed=df_tmp['event_status'],
                            label=i)
                    line_color = '#2878B5' if i == 'Low-risk' else '#c82423'
                    kmf.plot_survival_function(ci_show=False, ax=ax[nrow, ncol], color=line_color, show_censors=True)
                # Log-rank test
                p_value = multivariate_logrank_test(
                    event_durations=self.data['survival_time'],
                    groups=self.data[feature],
                    event_observed=self.data['event_status']
                ).p_value
                x += 1
                p_value_text = 'p-value < 0.001' if p_value < 0.001 else f'p-value = {p_value:.4F}'
                ax[nrow, ncol].set_title(f"{feature}", fontsize=8)
                ax[nrow, ncol].text(0.05, 0.08, p_value_text, fontsize=8, transform=ax[nrow][ncol].transAxes)
                ax[nrow, ncol].text(0.05, 0.04, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})",
                                    fontsize=8, transform=ax[nrow][ncol].transAxes)
                ax[nrow, ncol].set_xlabel("Days", fontsize=8)
                ax[nrow, ncol].set_ylabel("Proportion of Patients", fontsize=8)
                ax[nrow, ncol].tick_params(axis='x', labelsize=8)
                ax[nrow, ncol].tick_params(axis='y', labelsize=8)
                ax[nrow, ncol].legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def km_curve_every(self, num_row, num_col, save_dir='./visual_select_times'):
        """
        Plots and saves individual KM curves for each combination of feature selection and classifier methods.

        :param num_row: Number of rows in the grid.
        :param num_col: Number of columns in the grid.
        :param save_dir: Directory to save the plots.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        column_names = list(self.data.columns[0:-2])
        x = 0
        for nrow in range(num_row):
            for ncol in range(num_col):
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
                self.data[feature] = self.data[feature].replace({0: 'Low Risk', 1: 'High Risk'})
                for i in self.data[feature].unique():
                    kmf = KaplanMeierFitter()
                    df_tmp = self.data.loc[self.data[feature] == i]
                    kmf.fit(df_tmp['survival_time'],
                            event_observed=df_tmp['event_status'],
                            label=i)
                    line_color = '#2878B5' if i == 'Low Risk' else '#c82423'
                    kmf.plot_survival_function(ci_show=False, color=line_color, show_censors=True)
                p_value = multivariate_logrank_test(
                    event_durations=self.data['survival_time'],
                    groups=self.data[feature],
                    event_observed=self.data['event_status']
                ).p_value
                x += 1
                p_value_text = 'p-value < 0.001' if p_value < 0.001 else f'p-value = {p_value:.4F}'
                ax.set_title(f"{feature}", fontsize=12)
                ax.text(0.12, 0.17, p_value_text, fontsize=8, transform=plt.gcf().transFigure)
                ax.text(0.12, 0.14, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})",
                        fontsize=8, transform=plt.gcf().transFigure)
                ax.set_xlabel("Days", fontsize=12)
                ax.set_ylabel("Proportion of Patients", fontsize=12)
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                plt.tight_layout()
                plot_filename = f"{feats_selection_name}_{classifiers_name}.png"
                plt.savefig(os.path.join(save_dir, plot_filename))
                plt.close(fig)

    def km_curve_best_auc(self, best_feat_sel, best_classifier, save_path='./'):
        """
        Plots the KM curve for the best AUC combination of feature selection and classifier.

        :param best_feat_sel: The best feature selection method name as a string.
        :param best_classifier: The best classifier method name as a string.
        :param save_path: Path to save the KM curve plot.
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        best_col = self.data.filter(like=best_feat_sel).filter(like=best_classifier)
        if best_col.empty:
            print(f"No matching data found for feature selection '{best_feat_sel}' and classifier '{best_classifier}'.")
            return
        best_col_name = best_col.columns.tolist()[0]
        subset_data = self.data[['survival_time', 'event_status']]
        data = pd.concat([best_col, subset_data], axis=1)
        cph = CoxPHFitter()
        cph.fit(data, duration_col='survival_time', event_col='event_status')
        hr = cph.summary['exp(coef)'][0]
        ci_lower = cph.summary['exp(coef) lower 95%'][0]
        ci_higher = cph.summary['exp(coef) upper 95%'][0]
        best_col = best_col.replace({0: 'Low-risk', 1: 'High-risk'})
        best_col = best_col.squeeze()
        fig, ax = plt.subplots()
        n = 0
        for pre_type in best_col.unique():
            sub_data = self.data[best_col == pre_type]
            kmf = KaplanMeierFitter().fit(sub_data['survival_time'],
                                          event_observed=sub_data['event_status'],
                                          label=f'{pre_type}')
            line_color = '#2878B5' if pre_type == 'Low-risk' else '#c82423'
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
        ax.text(0.21, 0.4, f"HR:{hr:.2f} ({ci_lower:.2f}-{ci_higher:.2f})",
                fontsize=8, transform=plt.gcf().transFigure)
        ax.legend()
        ax.set_title(best_col_name)
        ax.set_xlabel('Days')
        ax.set_ylabel('Proportion of Patients')
        plt.tight_layout()
        plot_filename = f"KM_Curve_{best_col_name}.png"
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close(fig)
        print(f"KM curve saved to {os.path.join(save_path, plot_filename)}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Survival Analysis Plotter")
    parser.add_argument('--data_file', type=str, required=True,
                        help="Path to the CSV file containing prediction results and survival data.")
    parser.add_argument('--list_feats_selection', nargs='+',
                        default=['Univariate', 'mrmr', 'ttest', 'ranksums'],
                        help="List of feature selection methods.")
    parser.add_argument('--list_classifiers', nargs='+',
                        default=['QDA', 'LDA', 'RandomForest', 'LinearSVC', 'MLP', 'GaussianNB', 'AdaBoost'],
                        help="List of classifier methods.")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="Number of workers for parallel processing.")
    parser.add_argument('--save_path', type=str, default='./',
                        help="Path to save the plots.")
    parser.add_argument('--best_feat_sel', type=str, default='ranksums',
                        help="Best feature selection method name for plotting best AUC KM curve.")
    parser.add_argument('--best_classifier', type=str, default='RandomForest',
                        help="Best classifier method name for plotting best AUC KM curve.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    plotter = SurvivalAnalysisPlotter(
        data_file=args.data_file,
        list_feats_selection=args.list_feats_selection,
        list_classifiers=args.list_classifiers,
        n_workers=args.n_workers
    )
    # Plot KM curves for all combinations
    # plotter.km_curve_set(num_row=len(plotter.processed_feats_selection), num_col=len(plotter.processed_classifiers))
    # Plot individual KM curves and save them
    # plotter.km_curve_every(num_row=len(plotter.processed_feats_selection), num_col=len(plotter.processed_classifiers), save_dir=args.save_path)
    # Plot KM curve for the best AUC combination
    plotter.km_curve_best_auc(
        best_feat_sel=args.best_feat_sel,
        best_classifier=args.best_classifier,
        save_path=args.save_path
    )
