import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")


class DifferentialExpressionAnalyzer:
    """
    This class performs differential expression analysis between long-term and short-term groups
    based on texture features and gene expression data.
    """

    def __init__(
        self,
        feature_matrix_path: str,
        bulk_rna_path: str,
        save_path: str,
        top_features: List[str],
    ):
        """
        Initializes the analyzer with paths to data and the list of top features.

        :param feature_matrix_path: Path to the texture feature matrix CSV file.
        :param bulk_rna_path: Path to the bulk RNA data CSV file.
        :param save_path: Path where results will be saved.
        :param top_features: List of top features to analyze.
        """
        self.feature_matrix_path = feature_matrix_path
        self.bulk_rna_path = bulk_rna_path
        self.save_path = save_path
        self.top_features = top_features
        self.feature_matrix = None
        self.bulk_rna = None

    def load_data(self):
        """Loads the feature matrix and bulk RNA data."""
        self.feature_matrix = pd.read_csv(self.feature_matrix_path)
        self.bulk_rna = pd.read_csv(self.bulk_rna_path)
        print("Data loaded successfully.")

    def split_data_by_feature(self):
        """
        Splits the feature matrix into long-term and short-term groups based on the median
        of each feature in the top features list.
        """
        for feature in self.top_features:
            median_value = self.feature_matrix[feature].median()
            long_term_df = self.feature_matrix[
                self.feature_matrix[feature] <= median_value
            ]
            short_term_df = self.feature_matrix[
                self.feature_matrix[feature] > median_value
            ]

            # Create directories for saving data
            paths = {
                "long_term": os.path.join(self.save_path, feature, "long_term"),
                "short_term": os.path.join(self.save_path, feature, "short_term"),
                "result": os.path.join(self.save_path, feature, "result"),
            }
            for path in paths.values():
                os.makedirs(path, exist_ok=True)

            # Save the split data
            long_term_df.to_csv(
                os.path.join(paths["long_term"], f"long_term_{feature}.csv"),
                index=False,
            )
            short_term_df.to_csv(
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
            long_term_df.rename(
                columns={"patient_id": "patient_id"}, inplace=True
            )
            short_term_df.rename(
                columns={"patient_id": "patient_id"}, inplace=True
            )

            long_term_ids = long_term_df["patient_id"].str[:12]
            short_term_ids = short_term_df["patient_id"].str[:12]

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
            if sample[:12] in ids.values:
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
                log_fc =  mean_short - mean_long
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
            significant_genes = results_df[results_df["Q-Value"] <= 0.05]
            significant_genes.sort_values("Q-Value", inplace=True)

            # Save results
            significant_genes.to_excel(
                os.path.join(result_dir, "significant_genes.xlsx"),
                index=False,
            )
            print(
                f"Differential expression analysis completed for feature '{feature}'."
            )

    def run_analysis(self):
        """Runs the full analysis pipeline."""
        self.load_data()
        self.split_data_by_feature()
        self.extract_gene_expression_data()
        self.perform_differential_expression()
        print("Analysis complete.")

if __name__ == '__main__':
    # Replace with your actual paths and parameters
    feature_matrix_path = '/home/yuxin/bme/pypathomics/gene_output/different_expressed_genes/M_matrix_TCGA164.csv'
    save_path = '/home/yuxin/bme/pypathomics/gene_test/expression_analysis'
    top_features = ['glrlm_LongRunEmphasis_average_20']
    bulk_rna_path = '/home/yuxin/bme/pypathomics/gene_output/different_expressed_genes/RNA_delete_11_02_log(n+1)_transformed(fpkm).csv'

    # feature_matrix = '/path/to/feature_matrix.csv'
    # save_path = '/path/to/save_results'
    # top_features = ['glrlm_LongRunEmphasis_average_20']
    # bulk_rna = '/path/to/bulk_RNA.csv'

    # Initialize and run the analyzer
    analyzer = DifferentialExpressionAnalyzer(
        feature_matrix_path, bulk_rna_path, save_path, top_features
    )
    analyzer.run_analysis()
