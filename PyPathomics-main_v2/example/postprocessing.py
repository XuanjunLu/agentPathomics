import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS=None


def merge_wsi_feats(folder):
    csv_files = list(folder.glob("*.csv"))
    if len(csv_files) == 0:
        return
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def image_level_feats_aggregation(wsi_image_level_n, label_path):
    """
    The aggregation of image feats in one dataset

    :param wsi_image_level_n: str, all csv files of wsi_image_level_n
    :param label_path: str, labels of patients (0/1)

    :return: DataFrame, dataset feature matrix, each row represents a feature vector of a WSI, with the first
    column being the wsi_id
    """
    wsi_image_level_n = Path(wsi_image_level_n)
    label_path = Path(label_path)
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
            return None

    aggregation_matrix = pd.concat(aggregation_matrix, ignore_index=True)
    aggregation_matrix_label = pd.merge(aggregation_matrix, label, on="wsi_id", how="inner")
    # aggregation_matrix_label.to_csv("./11.csv", index=False)

    return aggregation_matrix_label


def nuclei_level_feats_aggregation(wsi_nuclei_level_n, label_path, nuclei_type=None):
    """
    The aggregation of nuclei feats in one dataset

    :param wsi_nuclei_level_n: str, all csv files of wsi_nuclei_level_n
    :param label_path: str, labels of patients (0/1)
    :param nuclei_type: integer, [1, 2, 3, 4, 5], (e.g., 1 is neoplastic nuclei, 2 is inflammatory nuclei), None
    gets all nuclei

    :return: DataFrame, dataset feature matrix, each row represents a feature vector of a WSI, with the first
    column being the wsi_id
    """
    wsi_image_level_n = Path(wsi_nuclei_level_n)
    label_path = Path(label_path)
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

    csv_files = list(wsi_image_level_n.glob("*.csv"))
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
            return None

    aggregation_matrix = pd.concat(aggregation_matrix, ignore_index=True)
    aggregation_matrix_label = pd.merge(aggregation_matrix, label, on="wsi_id", how="inner")
    # aggregation_matrix_label.to_csv("./11.csv", index=False)

    return aggregation_matrix_label



def merge_dataset_feature_matrix(wsi_image_level_n=None, wsi_nuclei_level_n=None, wsi_image_level_t=None,
                                 label_path=None, save_dir=None, raw_data=False):
    """
       Merge wsi-level feature to get the dataset-level feature matrix. If raw_data=True, return the extracted raw data;
    otherwise, return the feature matrix of the whole dataset after feature aggregation.

    :param wsi_image_level_n: str, the directory of nuclear image features in the WSI (csv file)
    :param wsi_nuclei_level_n: str, the directory of single nuclear features in the WSI (csv file)
    :param wsi_image_level_t: str, the directory of tissue features in the WSI (csv file)
    :param label_path: str, the directory of label(0/1) in the WSI
    :param save_dir: str, results saving path
    :param raw_data: bool, default False, not return raw data

    :return: DataFrame, the feature matrix of the whole dataset (all WSIs), each row represents a feature vector of a
    WSI, with the first column being the wsi_id and the last column being the label (0/1)

    """
    save_dir.mkdir(exist_ok=True)
    if raw_data:
        raw_data_dir = save_dir / 'raw_data'
        raw_data_dir.mkdir(exist_ok=True)
        r = 0
        if wsi_image_level_n.exists():
            df_image_level_n = merge_wsi_feats(wsi_image_level_n)
            if df_image_level_n is not None:
                df_image_level_n.to_csv(raw_data_dir / "raw_data_nuclei_image.csv", index=False)
                r += 1
        if wsi_nuclei_level_n.exists():
            df_nuclei_level_n = merge_wsi_feats(wsi_nuclei_level_n)
            if df_nuclei_level_n is not None:
                df_nuclei_level_n.to_csv(raw_data_dir / "raw_data_nuclei_nucleus.csv", index=False)
                r += 1
        if wsi_image_level_t.exists():
            df_image_level_t = merge_wsi_feats(wsi_image_level_t)
            if df_image_level_t is not None:
                df_image_level_t.to_csv(raw_data_dir / "raw_data_tissue_image.csv", index=False)
                r += 1

        if r == 0:
            warnings.warn("Neither nuclear features nor tissue features exist.")

        return

    else:
        dataset_feature_matrix = []
        if wsi_image_level_n.exists():
            aggregation_nuclei_image = image_level_feats_aggregation(wsi_image_level_n, label_path)
            if aggregation_nuclei_image is not None:
                aggregation_nuclei_image = aggregation_nuclei_image.add_suffix("_nuclei")
                dataset_feature_matrix.append(aggregation_nuclei_image)
        if wsi_nuclei_level_n.exists():
            aggregation_nuclei_nuclei = nuclei_level_feats_aggregation(wsi_nuclei_level_n, label_path, nuclei_type=None)
            if aggregation_nuclei_nuclei is not None:
                aggregation_nuclei_nuclei = aggregation_nuclei_nuclei.add_suffix("_nucleus")
                dataset_feature_matrix.append(aggregation_nuclei_nuclei)
        if wsi_image_level_t.exists():
            aggregation_tissue_image = image_level_feats_aggregation(wsi_image_level_t, label_path).add_suffix("_tissue")
            if aggregation_tissue_image is not None:
                aggregation_tissue_image = aggregation_tissue_image.add_suffix("_tissue")
                dataset_feature_matrix.append(aggregation_tissue_image)

        if dataset_feature_matrix:
            dataset_feature_matrix = pd.concat(dataset_feature_matrix, axis=1)
        else:
            raise ValueError("Feature is empty!")

        wsi_cols = [col for col in dataset_feature_matrix.columns if col.startswith('wsi')]
        label_cols = [col for col in dataset_feature_matrix.columns if col.startswith('label')]
        # startswith(wsi) columns
        if len(wsi_cols) == 1:
            dataset_feature_matrix = dataset_feature_matrix.rename(columns={wsi_cols[0]: 'wsi_id'})
        else:
            cols_to_drop = wsi_cols[1:]
            dataset_feature_matrix = dataset_feature_matrix.drop(columns=cols_to_drop)
            dataset_feature_matrix = dataset_feature_matrix.rename(columns={wsi_cols[0]: 'wsi_id'})

        # startswith(label) columns
        if len(label_cols) == 1:
            dataset_feature_matrix = dataset_feature_matrix.rename(columns={label_cols[0]: 'label'})
        else:
            cols_to_drop = label_cols[:-1]
            dataset_feature_matrix = dataset_feature_matrix.drop(columns=cols_to_drop)
            dataset_feature_matrix = dataset_feature_matrix.rename(columns={label_cols[-1]: 'label'})

        dataset_feature_matrix = (dataset_feature_matrix.replace([np.inf, -np.inf], np.nan)
                                  .fillna(dataset_feature_matrix.mean(numeric_only=True))
                                  .replace([np.inf, -np.inf, np.nan], 0))      # all nan
        dataset_feature_matrix.to_csv(f'{save_dir}/dataset_feature_matrix.csv', index=False)
        return dataset_feature_matrix


if __name__ == "__main__":
    wsi_image_level_n = Path("../../example_folder/nuclei_feats_extraction/wsi_image_level")
    wsi_nuclei_level_n = Path("../../example_folder/nuclei_feats_extraction/wsi_nuclei_level")
    wsi_image_level_t = Path("../../example_folder/tissue_feats_extraction/wsi_image_level")
    label_path = Path("../example_folder/survival_info.csv")
    save_dir = Path("../example_folder/aggregation")

    dataset_feature_matrix = merge_dataset_feature_matrix(wsi_image_level_n, wsi_nuclei_level_n, wsi_image_level_t,
                                                          label_path, save_dir, raw_data=False)
