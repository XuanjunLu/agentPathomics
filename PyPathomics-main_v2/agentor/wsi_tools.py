import os.path
import shutil
import time
import traceback
import uuid
from pathlib import Path
from langchain_core.tools import tool

from generate_pdf import generate_pdf_for_pathomics
from example.extract_wsi_feats_v4 import nuclei_feats_extraction, tissue_feats_extraction
from example.postprocessing import merge_dataset_feature_matrix
from pathomics.model_constructor.Btrain_test_cross_validationV3 import cross_validation
from pathomics.gene_path_analyzer.differential_expression_analyzerV2 import DifferentialExpressionAnalyzer
from pathomics.gene_path_analyzer.gene_enrichment_analyzerv3 import enrichment_analysis
"""
Note (Cursor project workaround):
If you hit JSON parsing issues in tool calls with escaped quotes, patch
`langchain_openai/chat_models/base.py::_convert_dict_to_message` to normalize
the `tool_calls[].function.arguments` string.
"""

# Global LLM configuration (can be set by GUI)
_LLM_CONFIG = {
    "api_key": None,
    "base_url": None,
    "report_model": "qwen-long"
}

def set_llm_config(api_key=None, base_url=None, report_model=None):
    """Set global LLM configuration for report generation."""
    if api_key is not None:
        _LLM_CONFIG["api_key"] = api_key
    if base_url is not None:
        _LLM_CONFIG["base_url"] = base_url
    if report_model is not None:
        _LLM_CONFIG["report_model"] = report_model


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_ROOT = _REPO_ROOT / "example_folder"


def _ensure_dir(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"{what} is not a directory: {p}")


def _ensure_file(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} does not exist: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"{what} is not a file: {p}")


def _mk_patient_id(patient_id: str = None) -> str:
    if patient_id and str(patient_id).strip():
        return str(patient_id).strip()
    # Keep the existing log style: patient_<ms>
    return f"patient_{int(time.time() * 1000)}"


def _patient_root(patient_id: str) -> Path:
    p = _EXAMPLE_ROOT / patient_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


@tool
def segment_image(image_path: str = None, seg_type: str = None, patient_id: str = None) -> str:
    """
    Segment WSIs in a folder (demo-mode implementation).
    :param image_path: WSI folder path [str]
    :param seg_type: segmentation type [str], one of: cell, tissue, all. Default: cell.
    :param patient_id: patient identifier [str]
    :return: segmentation output locations (string summary)
    """
    print(f"Running segmentation image_path: {image_path}; seg_type:{seg_type}")
    patient_id = _mk_patient_id(patient_id)
    patient_root = _patient_root(patient_id)

    # This repository does not ship a runnable segmentation model.
    # For the built-in demo data under `example_folder/`, we copy the example masks
    # into the patient folder as "segmentation outputs". For real data, integrate
    # an actual segmentation model/service.
    if image_path is None:
        raise ValueError("image_path cannot be empty")
    wsi_dir = Path(image_path)
    _ensure_dir(wsi_dir, "image_path (WSI directory)")

    seg_type = (seg_type or "cell").strip().lower()
    if seg_type not in ("cell", "tissue", "all"):
        seg_type = "cell"

    # Source masks (demo data)
    src_nuclei = _EXAMPLE_ROOT / "nuclei_seg"
    src_tumor = _EXAMPLE_ROOT / "tumor_seg"
    src_roi = _EXAMPLE_ROOT / "roi_seg"

    # Destination outputs (archived under patient folder)
    dst_nuclei = patient_root / "nuclei_seg"
    dst_tumor = patient_root / "tumor_seg"
    dst_roi = patient_root / "roi_seg"

    res_parts = []
    if seg_type in ("cell", "all"):
        _ensure_dir(src_nuclei, "example nuclei_seg")
        _copy_dir(src_nuclei, dst_nuclei)
        res_parts.append(f"Cell segmentation saved to: {dst_nuclei}")
    if seg_type in ("tissue", "all"):
        _ensure_dir(src_tumor, "example tumor_seg")
        _copy_dir(src_tumor, dst_tumor)
        res_parts.append(f"Tissue segmentation saved to: {dst_tumor}")
        # ROI masks (optional; used by tissue feature extraction)
        if src_roi.exists():
            _copy_dir(src_roi, dst_roi)
            res_parts.append(f"ROI masks saved to: {dst_roi}")

    res_parts.append(f"Patient ID: {patient_id}")
    res = ";".join(res_parts)
    print(f"segment_image:{res}")
    return res


@tool
def extract_features(swi_fold_path: str, label_path: str, patient_id: str = None, cell_fold_path: str = None,
                     tissue_fold_path: str = None, extract_type: str = None, mask_suffix="png",
                     feature_type="shape", nuclei_type=None, wsi_num=200, mag=40, n_workers=3) -> str:
    """
    Extract hand-crafted features from segmentation outputs and aggregate into a dataset feature matrix.
    :param swi_fold_path: WSI folder path
    :param label_path: survival/label file path
    :param patient_id: patient identifier
    :param cell_fold_path: cell segmentation folder (nuclei masks)
    :param tissue_fold_path: tissue segmentation folder (tumor/ROI masks)
    :param extract_type: feature extraction type, one of: cell, tissue, all. Default: cell.
    :param mask_suffix: str, the suffix of nuclear instance mask, classification mask and tissue (tumor) mask, default:"png"
    :param feature_type: str, "shape", "texture", "topology", "interplay" or "all", if "all", extract all features,default:"shape"
    :param nuclei_type: integer, [1, 2, 3, 4, 5]. nuclei type for  feature extraction, if None, get all nuclei
    :param wsi_num: integer, [0, +inf), the selected number of WSIs for feature extraction (e.g., 200),default:200
    :param mag: integer, target magnification for feature extraction (e.g., 40, 20, 10),default:40
    :param n_workers: integer, the number of processes,default:3
    :return: string summary including the output feature matrix path
    """
    print(f"Extracting features from: {swi_fold_path}")
    patient_id = _mk_patient_id(patient_id)
    patient_root = _patient_root(patient_id)

    wsi_dir = Path(swi_fold_path)
    _ensure_dir(wsi_dir, "swi_fold_path (WSI directory)")
    label_file = Path(label_path)
    _ensure_file(label_file, "label_path (survival/label file)")

    # Segmentation result directories (from segment_image output or user-selected paths)
    cell_dir = Path(cell_fold_path) if cell_fold_path else (patient_root / "nuclei_seg")
    tissue_dir = Path(tissue_fold_path) if tissue_fold_path else (patient_root / "tumor_seg")
    roi_dir = (patient_root / "roi_seg") if (patient_root / "roi_seg").exists() else tissue_dir

    # Output directories (archived under patient_id)
    cell_save_dir = patient_root / "nuclei_feats_extraction"
    tissu_save_dir = patient_root / "tissue_feats_extraction"
    agg_dir = patient_root / "aggregation"
    cell_save_dir.mkdir(parents=True, exist_ok=True)
    tissu_save_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    extract_type = (extract_type or "cell").strip().lower()
    if extract_type not in ("cell", "tissue", "all"):
        extract_type = "cell"

    # 1) Nuclei features (requires nuclei instance/class + tumor mask)
    if extract_type in ("cell", "all"):
        _ensure_dir(cell_dir, "cell_fold_path (nuclei_seg directory)")
        _ensure_dir(tissue_dir, "tissue_fold_path (tumor_seg directory)")
        nuclei_feats_extraction(
            wsi_dir=str(wsi_dir),
            nuclei_dir=str(cell_dir),
            tumor_dir=str(tissue_dir),
            save_dir=str(cell_save_dir),
            mask_suffix=mask_suffix,
            wsi_num=wsi_num,
            mag=mag,
            n_workers=n_workers,
            feature_type=feature_type,
            nuclei_type=nuclei_type,
        )

    # 2) Tissue features (requires ROI mask; demo may use roi_seg or tumor_seg)
    if extract_type in ("tissue", "all"):
        _ensure_dir(roi_dir, "roi_dir (ROI directory)")
        tissue_feats_extraction(
            wsi_dir=str(wsi_dir),
            roi_dir=str(roi_dir),
            save_dir=str(tissu_save_dir),
            mask_suffix=mask_suffix,
            wsi_num=wsi_num,
            mag=mag,
            n_workers=n_workers,
        )

    # 3) Aggregate into dataset_feature_matrix.csv
    wsi_image_level_n = cell_save_dir / "wsi_image_level"
    wsi_nuclei_level_n = cell_save_dir / "wsi_nuclei_level"
    wsi_image_level_t = tissu_save_dir / "wsi_image_level"
    merge_dataset_feature_matrix(
        wsi_image_level_n=wsi_image_level_n,
        wsi_nuclei_level_n=wsi_nuclei_level_n,
        wsi_image_level_t=wsi_image_level_t,
        label_path=label_file,
        save_dir=agg_dir,
        raw_data=False,
    )

    res = f"Feature extraction and aggregation completed; Patient ID: {patient_id}; Feature matrix: {agg_dir / 'dataset_feature_matrix.csv'}"
    print(f"extract_features res:{res}")
    return res


@tool
def build_model(extract_path: str, label_path: str, patient_id: str = None, top_feature_num: int = 6, k_fold: int = 5,
                feature_score_method: str = 'addone', var_thresh: int = 0, corr_threshold: float = 0.9,
                repeats_num: int = 100, list_feats_selection_args: list = None,
                list_classifers_args: list = None, n_workers: int = 10) -> str:
    """
    Build predictive models via cross-validation.
    :param
    :param extract_path: feature matrix path (CSV) or an aggregation directory containing dataset_feature_matrix.csv
    :param label_path: survival/label file path
    :param patient_id: patient identifier
    :param top_feature_num: integer, the number of top features,default:6
    :param k_fold: integer, the number of folds,default:5
    :param feature_score_method: str, the method of calculating feature score. "addone" or "weighted",default:"addone"
    :param var_thresh: float, default 0, remove the redundant features with variance, default:0
    :param corr_threshold: float, default 0.9, remove the redundant features with correlation,default:0.9
    :param repeats_num: integer, times to repeat the experiment,default:100
    :param list_feats_selection_args: list, 10 feature selection methods,['Lasso', 'XGBoost', 'RandomForest',
    'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest', 'ranksums', 'mutualInfo'],default:['Lasso', 'XGBoost']
    :param list_classifers_args: list, 11 classifiers, ['QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh',
    'LinearSVC', 'MLP', 'GaussianNB', 'SGD', 'SVC_rbf', 'AdaBoost'],default:['QDA', 'LDA', 'RandomForest']
    :param n_workers: the number of processes,default:3
    :return: string summary including the cross-validation output directory
    """
    print(f"Building model | input: {extract_path}")

    patient_id = _mk_patient_id(patient_id)
    patient_root = _patient_root(patient_id)
    extract_p = Path(extract_path).resolve()
    if extract_p.is_file() and extract_p.suffix.lower() == ".csv":
        # Allow passing a feature-matrix CSV directly (useful for standalone Model Constructor testing)
        dataset_feature_matrix = extract_p
    else:
        # Default convention: extract_path points to an aggregation directory
        dataset_feature_matrix = (extract_p / 'dataset_feature_matrix.csv').resolve()
    _ensure_file(dataset_feature_matrix, "dataset_feature_matrix (feature matrix)")

    survival_info_dir = Path(label_path).resolve()
    _ensure_file(survival_info_dir, "label_path (survival/label file)")
    save_dir = (patient_root / 'cross_validation').resolve()
    if not list_feats_selection_args:
        list_feats_selection_args = ['ranksums', 'mutualInfo']
    if not list_classifers_args:
        list_classifers_args = ['LinearSVC', 'LDA', 'RandomForest']
    cross_validation(dataset_feature_matrix, survival_info_dir, save_dir, top_feature_num=top_feature_num,
                     k_fold=k_fold, feature_score_method=feature_score_method, var_thresh=var_thresh,
                     corr_threshold=corr_threshold, repeats_num=repeats_num,
                     list_feats_selection_args=list_feats_selection_args,
                     list_classifers_args=list_classifers_args, n_workers=n_workers)

    print(f"Model outputs saved to: {save_dir}")
    return f"Model outputs saved to: {save_dir}"


@tool
def analyze_results(extract_path: str, bulk_rna_path: str, cross_validation_path: str, clinical_info_path: str,
                    patient_id: str = None, top: int = 12, top_signif: int = 2000, sources: str = 'GO:All',
                    clinical_col="T_stage", category=False) -> str:
    """
    Analyze model results: differential expression + enrichment analysis.
    :param extract_path: feature matrix path (CSV) or an aggregation directory containing dataset_feature_matrix.csv
    :param bulk_rna_path: str Path to the bulk RNA data (log2(n+1) normalized) CSV file.
    :param clinical_info_path: str Path to the clinical_information CSV file.
    :param patient_id: optional patient identifier; auto-generated if not provided
    :param cross_validation_path: cross-validation output directory
    :param top: integer take the first N characters of the "wsi_id" to ensure that the IDs in the
        dataset feature matrix are consistent with those in the bulk RNA. default:12
    :param top_signif: integer integer,(0, +inf), select the top genes with the highest statistical significance.default:2000
    :param sources: str or list, specifies the source databases for enrichment analysis (default is 'GO:All' for all GO terms).
        You can specify multiple sources separated by commas (e.g., 'GO:All,KEGG,REAC').
        Supported options include:
        - 'GO:All' for all GO terms
        - 'GO:MF' for Molecular Function
        - 'GO:CC' for Cellular Component
        - 'GO:BP' for Biological Process
        - 'KEGG' for Kyoto Encyclopedia of Genes and Genomes
        - 'REAC' for Reactome
        - 'WP' for WikiPathways
        - 'TF' for Transcription Factors (Transfac)
        - 'MIRNA' for miRNA interactions (miRTarBase)
        - 'HPA' for Human Protein Atlas
        - 'CORUM' for CORUM protein complexes
        - 'HP' for Human Phenotype Ontology
        default:"GO:All"
    :param clinical_col: str, the column name of clinical information.default:'T_stage'
    :param category: bool, the clinical information column is categorical variables or continuous variables?default:False
    :return: string summary including DEG and enrichment output directories
    """
    patient_id = _mk_patient_id(patient_id)
    patient_root = _patient_root(patient_id)

    # Allow passing a feature-matrix CSV directly (e.g., M_matrix_*.csv)
    extract_p = Path(extract_path).resolve()
    if extract_p.is_file() and extract_p.suffix.lower() == ".csv":
        feature_matrix_path = str(extract_p)
    else:
        feature_matrix_path = str((extract_p / 'dataset_feature_matrix.csv').resolve())
    _ensure_file(Path(feature_matrix_path), "dataset_feature_matrix (feature matrix)")

    save_path = str((patient_root / 'degs').resolve())
    _ensure_dir(Path(cross_validation_path).resolve(), "cross_validation_path (cross-validation output directory)")
    _ensure_file(Path(bulk_rna_path).resolve(), "bulk_rna_path (Bulk RNA)")
    _ensure_file(Path(clinical_info_path).resolve(), "clinical_info_path (clinical information)")
    gene_save_dir = str((patient_root / "FEA_GSEA").resolve())

    # Idempotency: if outputs already exist for the same patient_id, skip expensive recomputation
    degs_dir = Path(save_path)
    fea_dir = Path(gene_save_dir)
    degs_done = degs_dir.exists() and any(degs_dir.rglob("significant_genes.xlsx"))
    # Check for both FEA results AND GSEA prerank report (required for report generation)
    fea_done = (fea_dir.exists() and 
                any(fea_dir.rglob("GO_enrichment_results(PyPathomics_fpkm).csv")) and
                any(fea_dir.rglob("gseapy.gene_set.prerank.report.csv")))
    if degs_done and fea_done:
        msg = (
            f"DEG results: {save_path}; Enrichment results: {gene_save_dir}; "
            f"Detected existing outputs, skipping re-run."
        )
        print(msg)
        return msg

    # Normalize sources parameter: handle comma-separated string (e.g., "GO:All,KEGG")
    if isinstance(sources, str) and ',' in sources:
        # Split by comma and strip whitespace
        sources = [s.strip() for s in sources.split(',')]
    # If sources is still a string and equals 'GO:All', leave it as is (will be handled by performFEA)
    # Otherwise, if it's a list, keep it as a list

    if not degs_done:
        # DifferentialExpressionAnalyzer(V2) signature:
        # (feature_matrix_path, bulk_rna_path, save_path, cross_validation_path, top_signif)
        # Note: `top` (top_str) must be set via attribute, not passed into the constructor.
        analyzer = DifferentialExpressionAnalyzer(
            Path(feature_matrix_path),
            Path(bulk_rna_path),
            degs_dir,
            Path(cross_validation_path),
            int(top_signif),
        )
        analyzer.top_str = int(top) if top is not None else None
        analyzer.run_analysis()

    # sources = 'GO:All'
    print(f"enrichment_analysis params:save_path:{save_path}, gene_save_dir:{gene_save_dir}, sources:{sources},"
          f"clinical_info_path:{clinical_info_path},clinical_col:{clinical_col}, category:{category}")
    if not fea_done:
        # gene_enrichment_analyzerv3 uses clinical_information.suffix; must pass a Path object
        enrichment_analysis(
            save_path,
            gene_save_dir,
            sources,
            Path(clinical_info_path).resolve(),
            clinical_col,
            category,
        )
    print(f"DEG results: {save_path}; Enrichment results: {gene_save_dir}")
    return f"DEG results: {save_path}; Enrichment results: {gene_save_dir}"


@tool
def generate_report(differential_results_fold_path: str = None, gene_results_fold_path: str = None,
                   cross_validation_path: str = None) -> str:
    """
    Generate a pathology analysis report (PDF) based on DEG + enrichment outputs.
    :param
    differential_results_fold_path: DEG results directory (optional)
    gene_results_fold_path: enrichment results directory (FEA_GSEA)
    cross_validation_path: cross-validation results directory (optional; if not provided, will search relative to gene_results_fold_path)
    :return: string containing the report text and the PDF output path
    """
    from wsi_agent import parse_res, parse_res_v2
    try:
        print("Generating report")
        print(f"gene_results_fold_path:{gene_results_fold_path}")
        if not gene_results_fold_path:
            raise ValueError("gene_results_fold_path cannot be empty (FEA_GSEA directory)")
        gene_root = Path(gene_results_fold_path).resolve()
        _ensure_dir(gene_root, "gene_results_fold_path (FEA_GSEA directory)")

        # Compatibility: pick the first feature subdirectory (naming may differ across versions)
        feature_dirs = sorted([p for p in gene_root.iterdir() if p.is_dir()])
        if not feature_dirs:
            raise FileNotFoundError(f"FEA_GSEA directory is empty: {gene_root}")
        feature_dir = feature_dirs[0]

        # Compatibility: support multiple GSEA output folder names
        gsea_candidates = [
            feature_dir / "GSEA_PyPathomics_fpkm" / "gseapy.gene_set.prerank.report.csv",
            feature_dir / "GSEA(PyPathomics_fpkm)" / "gseapy.gene_set.prerank.report.csv",
        ]
        gsea_report = next((p for p in gsea_candidates if p.exists()), None)
        if gsea_report is None:
            # Fallback: recursively search under feature_dir
            matches = list(feature_dir.rglob("gseapy.gene_set.prerank.report.csv"))
            gsea_report = matches[0] if matches else None
        if gsea_report is None:
            raise FileNotFoundError(f"gseapy.gene_set.prerank.report.csv not found (feature={feature_dir.name})")

        # Stats text: prefer legacy naming; otherwise use *_statistics.txt
        txt_candidates = [
            feature_dir / "T_stage_statistics.txt",
        ]
        stat_txt = next((p for p in txt_candidates if p.exists()), None)
        if stat_txt is None:
            matches = sorted(feature_dir.glob("*_statistics.txt"))
            stat_txt = matches[0] if matches else None

        res = parse_res_v2(
            str(gsea_report),
            txt_path=str(stat_txt) if stat_txt else None,
            api_key=_LLM_CONFIG.get("api_key"),
            base_url=_LLM_CONFIG.get("base_url"),
            report_model=_LLM_CONFIG.get("report_model", "qwen-long")
        )
        base_fold = str(gene_root.parent)
        
        # Locate cross_validation directory
        if cross_validation_path:
            cv_dir = Path(cross_validation_path).resolve()
        else:
            # Try to find cross_validation in the current patient directory first
            cv_dir = gene_root.parent / "cross_validation"
            if not cv_dir.exists() or not (cv_dir / "auc_table.csv").exists():
                # Search in sibling patient directories or parent example_folder
                search_root = gene_root.parent.parent  # example_folder level
                cv_candidates = list(search_root.rglob("cross_validation/auc_table.csv"))
                if cv_candidates:
                    cv_dir = cv_candidates[0].parent
                    print(f"Found cross_validation directory: {cv_dir}")
                else:
                    raise FileNotFoundError(
                        f"Cannot locate cross_validation/auc_table.csv. "
                        f"Searched in: {gene_root.parent} and {search_root}. "
                        f"Please provide cross_validation_path explicitly."
                    )
        
        _ensure_file(cv_dir / "auc_table.csv", "auc_table.csv (cross-validation results)")
        
        pdf_path = os.path.join(base_fold, f"{uuid.uuid1()}.pdf")
        generate_pdf_for_pathomics(base_fold, llm_res=res, out_put=pdf_path, cross_validation_dir=str(cv_dir))
        return f"Report text (verbatim): {res}\n\nPDF report saved to: {pdf_path}"
    except Exception as e:
        # Return the real Python exception to the GUI to avoid misleading "path not found" messages.
        return f"Report generation failed: {type(e).__name__}: {e}\n\nTraceback:\n{traceback.format_exc()}"
