import os
import argparse
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class DifferentialExpressionAnalyzer:
    """
    This class performs differential expression analysis between long-term and short-term groups
    based on texture features and gene expression data (bulk RNA).

    Args:
        :param feature_matrix_path: Path to the texture feature matrix CSV file.
        :param bulk_rna_path: Path to the bulk RNA data (log2(n+1) normalized) CSV file.
        :param save_path: Path where results will be saved.
        :param cross_validation_path: Path to the cross validation results
        :param top_str: integer, default None. take the first N characters of the "wsi_id" to ensure that the IDs in the
        dataset feature matrix are consistent with those in the bulk RNA.
        :param top_signif: integer,(0, +inf), select the top genes with the highest statistical significance

    Returns:
        Core output, list of differentially expressed genes (significant_genes.xlsx).
    """

    def __init__(
        self,
        feature_matrix_path: Path,
        bulk_rna_path: Path,
        save_path: Path,
        cross_validation_path: Path,
        top_signif: int
    ):
        """
        Initializes the analyzer with paths to data and the list of top features.
        """
        self.feature_matrix_path = feature_matrix_path
        self.bulk_rna_path = bulk_rna_path
        self.save_path = save_path
        self.top_features = None
        self.cross_validation_path = cross_validation_path
        self.feature_matrix = None
        self.bulk_rna = None
        self.top_str = None
        self.top_signif = top_signif
        self.long_short_dominant = None

    def load_data(self):
        """Loads the feature matrix and bulk RNA data."""
        self.feature_matrix = pd.read_csv(self.feature_matrix_path)
        self.bulk_rna = pd.read_csv(self.bulk_rna_path)

    def get_top1_feature(self):
        auc_tabel = self.cross_validation_path / "auc_table.csv"
        topk_feature = self.cross_validation_path / "topk_features.csv"
        auc_tabel = pd.read_csv(auc_tabel)
        topk_feature = pd.read_csv(topk_feature)

        # handle auc
        auc_tabel.rename(columns={'Unnamed: 0': 'selection'}, inplace=True)
        auc_tabel.set_index('selection', inplace=True)
        auc_tabel_std = auc_tabel.copy()
        for col in auc_tabel_std.columns:
            auc_tabel_std[col] = auc_tabel_std[col].apply(lambda x: float(x.split('/')[1]))
        for col in auc_tabel.columns:
            auc_tabel[col] = auc_tabel[col].apply(lambda x: float(x.split('/')[0]))

        # handle feature scores
        topk_feature.rename(columns={'Unnamed: 0': 'selection'}, inplace=True)
        topk_feature.set_index('selection', inplace=True)
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

        feature_list = []
        feature_names = topk_feature.loc[max_row, max_col]
        feature_names = eval(feature_names.replace(' ', ''))
        for i in range(1):              # len(feature_names)
            feature_list.append(feature_names[i])
        self.top_features = feature_list


    def split_data_by_feature(self):
        """
        Splits the feature matrix into long-term and short-term groups based on the median
        of each feature in the top features list.
        """
        for feature in self.top_features:
            median_value = self.feature_matrix[feature].median()
            group_low_df = self.feature_matrix[
                self.feature_matrix[feature] <= median_value
            ]
            group_high_df = self.feature_matrix[
                self.feature_matrix[feature] > median_value
            ]

            # the average value of label(0/1) in long term samples
            group_low_df_mean_label = group_low_df["label"].mean()
            # the average value of label(0/1) in short term samples
            group_high_df_mean_label = group_high_df["label"].mean()

            # long term samples are more alive, control group
            if group_low_df_mean_label <= group_high_df_mean_label:
                self.long_short_dominant = True
            # short term samples are more alive, control group
            else:
                self.long_short_dominant = False

            # Create directories for saving data
            paths = {
                "long_term": os.path.join(self.save_path, feature, "long_term"),
                "short_term": os.path.join(self.save_path, feature, "short_term"),
                "result": os.path.join(self.save_path, feature, "result"),
            }
            for path in paths.values():
                os.makedirs(path, exist_ok=True)

            if self.long_short_dominant:
                # Save the split data
                group_low_df.to_csv(
                    os.path.join(paths["long_term"], f"long_term_{feature}.csv"),
                    index=False,
                )
                group_high_df.to_csv(
                    os.path.join(paths["short_term"], f"short_term_{feature}.csv"),
                    index=False,
                )
            else:
                group_high_df.to_csv(
                    os.path.join(paths["long_term"], f"long_term_{feature}.csv"),
                    index=False,
                )
                group_low_df.to_csv(
                    os.path.join(paths["short_term"], f"short_term_{feature}.csv"),
                    index=False,
                )

            print(f"Data split by feature '{feature}' and saved.")

    def extract_gene_expression_data(self):
        """
        Extracts gene expression data corresponding to the patient IDs in the long-term and
        short-term groups.
        """
        for feature in tqdm(self.top_features, desc="Extracting gene data"):
            base_dir = os.path.join(self.save_path, feature)
            long_term_df = pd.read_csv(
                os.path.join(base_dir, "long_term", f"long_term_{feature}.csv")
            )
            short_term_df = pd.read_csv(
                os.path.join(base_dir, "short_term", f"short_term_{feature}.csv")
            )

            # Ensure patient_id column is correctly named
            # long_term_df.rename(
            #     columns={"patient_id": "patient_id"}, inplace=True
            # )
            # short_term_df.rename(
            #     columns={"patient_id": "patient_id"}, inplace=True
            # )
            if self.top_str is None:
                self.top_str = int(len(long_term_df["wsi_id"][0]))
            else:
                self.top_str = int(self.top_str)
            long_term_ids = long_term_df["wsi_id"].str[:self.top_str]
            short_term_ids = short_term_df["wsi_id"].str[:self.top_str]

            # Save patient IDs
            result_dir = os.path.join(base_dir, "result")
            long_term_ids.to_csv(
                os.path.join(result_dir, "long_term_names.txt"),
                index=False,
                header=False,
            )
            short_term_ids.to_csv(
                os.path.join(result_dir, "short_term_names.txt"),
                index=False,
                header=False,
            )

            # Extract gene expression data
            long_term_samples = self._match_samples(long_term_ids)
            short_term_samples = self._match_samples(short_term_ids)

            long_term_data = self.bulk_rna[long_term_samples]
            short_term_data = self.bulk_rna[short_term_samples]

            # Save extracted data
            with pd.ExcelWriter(
                os.path.join(result_dir, "extract_data.xlsx")
            ) as writer:
                long_term_data.to_excel(
                    writer, sheet_name="long_term", index=False
                )
                short_term_data.to_excel(
                    writer, sheet_name="short_term", index=False
                )

            # Save gene IDs
            gene_ids = self.bulk_rna.iloc[:, 0]
            gene_ids.to_excel(
                os.path.join(result_dir, "gene_id.xlsx"), index=False
            )
            print(f"Gene expression data extracted for feature '{feature}'.")

    def _match_samples(self, ids):
        """
        Matches sample IDs in the bulk RNA data with patient IDs.

        :param ids: Series of patient IDs.
        :return: List of matching sample IDs.
        """
        matched_samples = []
        for sample in self.bulk_rna.columns[1:]:
            if sample[:self.top_str] in ids.values:
                matched_samples.append(sample)
        return matched_samples

    def perform_differential_expression(self):
        """
        Performs differential expression analysis between long-term and short-term groups.
        """
        for feature in tqdm(self.top_features, desc="Performing DE analysis"):
            result_dir = os.path.join(self.save_path, feature, "result")
            # Load data
            long_term_data = pd.read_excel(
                os.path.join(result_dir, "extract_data.xlsx"),
                sheet_name="long_term",
            )
            short_term_data = pd.read_excel(
                os.path.join(result_dir, "extract_data.xlsx"),
                sheet_name="short_term",
            )
            gene_ids = pd.read_excel(
                os.path.join(result_dir, "gene_id.xlsx")
            )
            p_values = []
            log_fc_values = []
            # Perform statistical tests for each gene
            for idx in range(len(long_term_data)):
                long_gene = long_term_data.iloc[idx, :]
                short_gene = short_term_data.iloc[idx, :]

                # Calculate mean expressions
                mean_long = long_gene.mean()
                mean_short = short_gene.mean()

                # Calculate log fold change
                log_fc = mean_short - mean_long
                log_fc_values.append(log_fc)

                # Calculate p-value
                _, p_value = ranksums(long_gene, short_gene)
                p_values.append(p_value)

            # Adjust p-values using FDR
            _, q_values, _, _ = multipletests(
                p_values, alpha=0.05, method="fdr_bh"
            )

            # Compile results
            results_df = pd.DataFrame({
                "Gene ID": gene_ids.squeeze(),
                "LogFC": log_fc_values,
                "P-Value": p_values,
                "Q-Value": q_values,
            })

            # Filter significant genes
            # significant_genes = results_df[results_df["Q-Value"] <= 0.05]
            significant_genes = results_df
            significant_genes.sort_values("Q-Value", inplace=True)
            significant_genes = significant_genes.head(self.top_signif)

            # Save results
            significant_genes.to_excel(
                os.path.join(result_dir, "significant_genes.xlsx"),
                index=False)
            print(
                f"Differential expression analysis completed for feature '{feature}'."
            )

    def run_analysis(self):
        """Runs the full analysis pipeline."""
        self.load_data()
        self.get_top1_feature()
        self.split_data_by_feature()
        self.extract_gene_expression_data()
        self.perform_differential_expression()


def main():
    parser = argparse.ArgumentParser(description="Differential expression gene analysis")
    parser.add_argument("--dataset_feature_matrix", default="../../example_folder/aggregation/dataset_feature_matrix.csv", type=Path, required=True, help="Path to the feature matrix")
    parser.add_argument("--bulk_rna", default="../../example_folder/bulkRNA_delete_11_02_log(n+1)_transformed(fpkm)_rename.csv", type=Path, required=True, help="Path to the bulk RNA")
    parser.add_argument("--deg_save_dir", default="../../example_folder/Degs", type=Path, required=True, help="Path of save directory")
    parser.add_argument("--cross_validation", default="../../example_folder/cross_validation", type=Path, required=True, help="Path to cross-validation result")
    parser.add_argument("--top_signif", default=2000, type=int, required=True, help="The number of significant genes")
    args = parser.parse_args()
    analyzer = DifferentialExpressionAnalyzer(args.dataset_feature_matrix, args.bulk_rna, args.deg_save_dir, args.cross_validation, args.top_signif)
    analyzer.run_analysis()



if __name__ == '__main__':
    main()
