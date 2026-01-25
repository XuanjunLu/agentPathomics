import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import os
Image.MAX_IMAGE_PIXELS=None


def merge_wsi_feats(folder):
    csv_files = list(folder.glob("*.csv"))
    if len(csv_files) == 0:
        return
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs).reset_index(drop=True)
    return df

# def duplicate_cols_processing(dataset_feature_matrix):
#     """
#
#     :param dataset_feature_matrix: DataFrame, dataset_feature_matrix
#
#     :return: DataFrame, dataset feature matrix, no duplicated columns
#     """
#     wsi_cols = [col for col in dataset_feature_matrix.columns if col.startswith('wsi')]
#     label_cols = [col for col in dataset_feature_matrix.columns if col.startswith('label')]
#     # startswith(wsi) columns
#     if len(wsi_cols) == 1:
#         dataset_feature_matrix = dataset_feature_matrix.rename(columns={wsi_cols[0]: 'wsi_id'})
#     else:
#         cols_to_drop = wsi_cols[1:]
#         dataset_feature_matrix = dataset_feature_matrix.drop(columns=cols_to_drop)
#         dataset_feature_matrix = dataset_feature_matrix.rename(columns={wsi_cols[0]: 'wsi_id'})
#
#     # startswith(label) columns
#     if len(label_cols) == 1:
#         dataset_feature_matrix = dataset_feature_matrix.rename(columns={label_cols[0]: 'label'})
#     else:
#         cols_to_drop = label_cols[:-1]
#         dataset_feature_matrix = dataset_feature_matrix.drop(columns=cols_to_drop)
#         dataset_feature_matrix = dataset_feature_matrix.rename(columns={label_cols[-1]: 'label'})
#
#     dataset_feature_matrix = (dataset_feature_matrix.replace([np.inf, -np.inf], np.nan)
#                               .fillna(dataset_feature_matrix.mean(numeric_only=True))
#                               .replace([np.inf, -np.inf, np.nan], 0))  # all nan
#
#     return dataset_feature_matrix


def duplicate_2cols_processing(dataset_feature_matrix, index=1, col1=None, col2=None):
    """
    Delete duplicated id columns

    :param dataset_feature_matrix: DataFrame, dataset_feature_matrix
    :param index: integer, 1 or -1, needed column index
    :param col1: str, column name
    :param col2: str, column name

    :return: DataFrame, dataset feature matrix, no duplicated columns
    """
    # startswith(wsi) columns
    wsi_id_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col1)]
    if len(wsi_id_cols) == 1:
        pass
    else:
        cols_to_drop_index = wsi_id_cols[1:]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    # startswith(label) columns
    image_id_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col2)]
    if len(image_id_cols) == 1:
        pass
    elif index == 1:
        cols_to_drop_index = image_id_cols[index:]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]
    else:
        cols_to_drop_index = image_id_cols[:index]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]
    # deal with NA
    dataset_feature_matrix = (dataset_feature_matrix.replace([np.inf, -np.inf], np.nan)
                              .fillna(dataset_feature_matrix.mean(numeric_only=True))
                              .replace([np.inf, -np.inf, np.nan], 0))  # all nan

    return dataset_feature_matrix



def duplicate_5cols_processing(dataset_feature_matrix, col1=None, col2=None, col3=None, col4=None, col5=None):
    """
    Delete duplicated id columns

    :param dataset_feature_matrix: DataFrame, dataset_feature_matrix
    :param col1: str, column name
    :param col2: str, column name
    :param col3: str, column name
    :param col4: str, column name
    :param col5: str, column name

    :return: DataFrame, dataset feature matrix, no duplicated columns
    """
    wsi_id_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col1)]
    # startswith(wsi) columns
    if len(wsi_id_cols) == 1:
        pass
    else:
        cols_to_drop_index = wsi_id_cols[1:]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    # startswith(image) columns
    image_id_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col2)]
    if len(image_id_cols) == 1:
        pass
    else:
        cols_to_drop_index = image_id_cols[1:]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    # startswith(nuclei_id) columns
    nuclei_id_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col3)]
    if len(nuclei_id_cols) == 1:
        pass
    else:
        cols_to_drop_index = nuclei_id_cols[1:]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    # startswith(tumor_status) columns
    tumor_status_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col4)]
    if len(tumor_status_cols) == 1:
        pass
    else:
        cols_to_drop_index = tumor_status_cols[:-1]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    # startswith(nuclei_class) columns
    nuclei_class_cols = [i for i, col in enumerate(dataset_feature_matrix.columns) if col.startswith(col5)]
    if len(nuclei_class_cols) == 1:
        pass
    else:
        cols_to_drop_index = nuclei_class_cols[:-1]
        keep_index = [idx for idx in range(len(dataset_feature_matrix.columns)) if idx not in cols_to_drop_index]
        dataset_feature_matrix = dataset_feature_matrix.iloc[:, keep_index]

    dataset_feature_matrix = (dataset_feature_matrix.replace([np.inf, -np.inf], np.nan)
                              .fillna(dataset_feature_matrix.mean(numeric_only=True))
                              .replace([np.inf, -np.inf, np.nan], 0))  # all nan

    return dataset_feature_matrix





def image_level_feats_aggregation(wsi_image_level_n: Path, label_path: Path):
    """
    The aggregation of image feats in one dataset

    :param wsi_image_level_n: str, all csv files of wsi_image_level_n
    :param label_path: str, labels of patients (0/1)

    :return: DataFrame, dataset feature matrix, each row represents a feature vector of a WSI, with the first
    column being the wsi_id
    """
    label_suffix = label_path.suffix
    if label_suffix == '.csv':
        label = pd.read_csv(label_path)
    elif label_suffix in ['.xlsx', '.xls']:
        label = pd.read_excel(label_path)
    else:
        raise TypeError(f"Supported input formats: .csv, .xlsx, .xls")
    required_columns = ['wsi_id', 'label']
    for col in required_columns:
        if col not in label.columns.to_list():
            raise ValueError(f'Please ensure that two columns: "wsi_id" and "label".')
    label = label[required_columns]

    csv_files = list(wsi_image_level_n.glob("*.csv"))
    if len(csv_files) == 0:
        return
    aggregation_matrix = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            wsi_id = df["wsi_id"][0]
            df.drop(columns=["wsi_id", "image_id"], inplace=True)
            df_mean = df.mean(axis=0)
            df_mean = pd.DataFrame([df_mean])
            df_mean.columns = df.columns
            df_mean.insert(0, "wsi_id", wsi_id)
            aggregation_matrix.append(df_mean)
        else:
            return

    aggregation_matrix = pd.concat(aggregation_matrix, ignore_index=True)
    aggregation_matrix_label = pd.merge(aggregation_matrix, label, on="wsi_id", how="inner")
    aggregation_matrix_label = aggregation_matrix_label.set_index("wsi_id")
    # aggregation_matrix_label.to_csv("./11.csv", index=False)

    return aggregation_matrix_label


def nuclei_level_feats_aggregation(wsi_nuclei_level_n: Path, label_path: Path, nuclei_type=None):
    """
    The aggregation of nuclei feats in one dataset

    :param wsi_nuclei_level_n: str, all csv files of wsi_nuclei_level_n
    :param label_path: str, labels of patients (0/1)
    :param nuclei_type: integer, [1, 2, 3, 4, 5], (e.g., 1 is neoplastic nuclei, 2 is inflammatory nuclei), None
    gets all nuclei

    :return: DataFrame, dataset feature matrix, each row represents a feature vector of a WSI, with the first
    column being the wsi_id
    """

    label_suffix = label_path.suffix
    if label_suffix == '.csv':
        label = pd.read_csv(label_path)
    elif label_suffix in ['.xlsx', '.xls']:
        label = pd.read_excel(label_path)
    else:
        raise ValueError(f"Supported input formats: .csv, .xlsx, .xls")
    required_columns = ['wsi_id', 'label']
    for col in required_columns:
        if col not in label.columns.to_list():
            raise ValueError(f'Please ensure that two columns: "wsi_id" and "label".')
    label = label[required_columns]

    csv_files = list(wsi_nuclei_level_n.glob("*.csv"))
    if len(csv_files) == 0:
        return
    aggregation_matrix = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            if nuclei_type is not None:
                df = df[(df["tumor_status"] == 1) & (df["nuclei_class"] == nuclei_type)]
            else:
                df = df[df["tumor_status"] == 1]
            df = df.reset_index(drop=True)
            wsi_id = df["wsi_id"][0]
            df.drop(columns=["wsi_id", "image_id", "nuclei_id", "tumor_status", "nuclei_class"], inplace=True)
            df_mean = df.mean(axis=0)
            df_mean = pd.DataFrame([df_mean])
            df_mean.columns = df.columns
            df_mean.insert(0, "wsi_id", wsi_id)
            aggregation_matrix.append(df_mean)
        else:
            return

    aggregation_matrix = pd.concat(aggregation_matrix, ignore_index=True)
    aggregation_matrix_label = pd.merge(aggregation_matrix, label, on="wsi_id", how="inner")
    aggregation_matrix_label = aggregation_matrix_label.set_index("wsi_id")
    # aggregation_matrix_label.to_csv("./11.csv", index=False)

    return aggregation_matrix_label



def merge_dataset_feature_matrix(nuclei_feats_extraction: Path, tissue_feats_extraction: Path,
                                 label_path: Path, save_dir: Path, raw_data=False):
    """
       Merge wsi-level feature to get the dataset-level feature matrix. If raw_data=True, return the extracted raw data;
    otherwise, return the feature matrix of the whole dataset after feature aggregation.

    :param nuclei_feats_extraction: str, the directory of nuclear features in the WSI (csv file)
    :param tissue_feats_extraction: str, the directory of tissue features in the WSI (csv file)
    :param label_path: str, the directory of label(0/1) in the WSI
    :param save_dir: str, results saving path
    :param raw_data: bool, default False, not return raw data

    :return: DataFrame, the feature matrix of the whole dataset (all WSIs), each row represents a feature vector of a
    WSI, with the first column being the wsi_id and the last column being the label (0/1)

    """
    save_dir.mkdir(exist_ok=True)
    nuclei_feats_folders = nuclei_feats_extraction.glob('*')       # 1 folders
    tissue_feats_folders = tissue_feats_extraction.glob('*')       # 6 folders
    mitosis_count_file = os.path.join(os.path.dirname(tissue_feats_extraction), "mitosis_count", "mitosis_count.csv")
    mitosis_count_file = pd.read_csv(mitosis_count_file)

    if raw_data:
        raw_data_dir = save_dir / 'raw_data'
        raw_data_dir.mkdir(exist_ok=True)
        r = 0
        df_image_level_n = []
        df_nuclei_level_n = []
        df_image_level_t = []
        for nuclei_feats_folder in nuclei_feats_folders:
            wsi_image_level_n = nuclei_feats_folder / "wsi_image_level"
            wsi_nuclei_level_n = nuclei_feats_folder / "wsi_nuclei_level"
            if wsi_image_level_n.exists():
                merge_image_level_n = merge_wsi_feats(wsi_image_level_n)
                if merge_image_level_n is not None:
                    df_image_level_n.append(merge_image_level_n)
                    # if missing values, filled with NA
            if wsi_nuclei_level_n.exists():
                merge_nuclei_level_n = merge_wsi_feats(wsi_nuclei_level_n)
                if merge_nuclei_level_n is not None:
                    df_nuclei_level_n.append(merge_nuclei_level_n)
                    # if missing values, filled with NA
        df_image_level_n = pd.concat(df_image_level_n, axis=1)
        df_image_level_n = duplicate_2cols_processing(df_image_level_n, col1="wsi_id", col2="image_id")
        df_image_level_n.to_csv(raw_data_dir / "raw_data_nuclei_image.csv", index=False)
        df_nuclei_level_n = pd.concat(df_nuclei_level_n, axis=1)
        df_nuclei_level_n = duplicate_5cols_processing(df_nuclei_level_n, col1="wsi_id", col2="image_id",
                                                       col3="nuclei_id", col4="tumor_status", col5="nuclei_class")
        df_nuclei_level_n.to_csv(raw_data_dir / "raw_data_nuclei_nucleus.csv", index=False)
        mitosis_count_file.to_csv(raw_data_dir / "raw_data_mitosis_count.csv", index=False)
        r += 1

        for tissue_feats_folder in tissue_feats_folders:
            wsi_image_level_t = tissue_feats_folder / "wsi_image_level"
            if wsi_image_level_t.exists():
                merge_image_level_t = merge_wsi_feats(wsi_image_level_t)
                if merge_image_level_t is not None:
                    df_image_level_t.append(merge_image_level_t)
                    # if missing values, filled with NA
        df_image_level_t = pd.concat(df_image_level_t, axis=1)
        df_image_level_t.to_csv(raw_data_dir / "raw_data_tissue_image.csv", index=False)
        r += 1
        if r == 0:
            warnings.warn("Neither nuclear features nor tissue features exist.")

        return

    else:
        dataset_feature_matrix = []
        for nuclei_feats_folder in nuclei_feats_folders:            # nuclei/mitosis
            wsi_image_level_n = nuclei_feats_folder / "wsi_image_level"
            wsi_nuclei_level_n = nuclei_feats_folder / "wsi_nuclei_level"
            if wsi_image_level_n.exists():
                aggregation_nuclei_image = image_level_feats_aggregation(wsi_image_level_n, label_path)
                if aggregation_nuclei_image is not None:
                    dataset_feature_matrix.append(aggregation_nuclei_image)
            if wsi_nuclei_level_n.exists():
                aggregation_nuclei_nuclei = nuclei_level_feats_aggregation(wsi_nuclei_level_n, label_path, nuclei_type=None)
                if aggregation_nuclei_nuclei is not None:
                    dataset_feature_matrix.append(aggregation_nuclei_nuclei)
        for tissue_feats_folder in tissue_feats_folders:
            wsi_image_level_t = tissue_feats_folder / "wsi_image_level"
            if wsi_image_level_t.exists():
                aggregation_tissue_image = image_level_feats_aggregation(wsi_image_level_t, label_path)
                if aggregation_tissue_image is not None:
                    dataset_feature_matrix.append(aggregation_tissue_image)

        if dataset_feature_matrix:
            dataset_feature_matrix = pd.concat(dataset_feature_matrix, axis=1, join="inner")
            dataset_feature_matrix = dataset_feature_matrix.reset_index()
        else:
            raise ValueError("Feature is empty!")

        dataset_feature_matrix = duplicate_2cols_processing(dataset_feature_matrix, index=-1, col1="wsi_id", col2="label")
        wsi_id_cols = [col for col in dataset_feature_matrix.columns if col.startswith("wsi_id")]
        label_cols = [col for col in dataset_feature_matrix.columns if col.startswith("label")]
        dataset_feature_matrix = dataset_feature_matrix.rename(columns={wsi_id_cols[0]: 'wsi_id'})
        dataset_feature_matrix = dataset_feature_matrix.rename(columns={label_cols[0]: 'label'})
        dataset_feature_matrix = pd.merge(dataset_feature_matrix, mitosis_count_file, on="wsi_id", how="inner")
        dataset_feature_matrix.to_csv(f'{save_dir}/dataset_feature_matrix.csv', index=False)
        failed_sample = int(len(aggregation_nuclei_nuclei) - len(dataset_feature_matrix))

        return dataset_feature_matrix, failed_sample


def main():
    parser = argparse.ArgumentParser(description="Postprocessing: merge features")
    parser.add_argument("--nuclei_feats_extraction",default="../example_folder/nuclei_feats_extraction",  type=Path, required=True, help="Feature extraction results of nuclei")
    parser.add_argument("--tissue_feats_extraction", default="../example_folder/tissue_feats_extraction", type=Path, required=True, help="Feature extraction results of tissue")
    parser.add_argument("--label_path", default="../example_folder/survival_info.csv", type=Path, required=True, help="Path of label file")
    parser.add_argument("--post_save_dir", default="../example_folder/aggregation", type=Path, required=True, help="Path of save directory")
    args = parser.parse_args()
    print("Postprocessing: start merging features...")
    _, failed_sample = merge_dataset_feature_matrix(args.nuclei_feats_extraction, args.tissue_feats_extraction,
                                                    args.label_path, args.post_save_dir, raw_data=False)
    if failed_sample > 0:
        print(f"{failed_sample} samples are excluded due to insufficient tissue size.")


if __name__ == "__main__":
    main()
