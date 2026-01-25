#!/bin/bash

##### Pipeline: 1→2→3→4→5→6
set -Eeuo pipefail

# ===================== Error handling =====================
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +'%Y%m%d_%H%M%S').log"

exec > >(tee -a "$LOG_FILE") 2>&1

err_report() {
  local exit_code=$?
  local line_no=$1
  local cmd=${BASH_COMMAND:-"UNKNOWN"}
  echo
  echo "==================== PIPELINE FAILED ===================="
  echo "Exit code : $exit_code"
  echo "Line      : $line_no"
  echo "Command   : $cmd"
  echo "Log file  : $LOG_FILE"
  echo "========================================================="
  exit "$exit_code"
}
trap 'err_report $LINENO' ERR
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Start time: $(date)"



#===================== Customize DIR=====================
WSI_DIR="../example_folder/svs"
LABEL_PATH="../example_folder/survival_info.csv"
BULK_RNA="../../example_folder/bulkRNA_delete_11_02_log(n+1)_transformed(fpkm)_rename.csv"
PROXY_PORT=7897


#================= Not change parameters =================
NUCLEI_DIR="$(dirname "${WSI_DIR}")/nuclei_seg"
TUMOR_DIR="$(dirname "${WSI_DIR}")/tissue_seg/tumor"
NUCLEI_SAVE_DIR="$(dirname "${WSI_DIR}")/nuclei_feats_extraction"
TISSUE_ROI_DIR="$(dirname "${WSI_DIR}")/tissue_seg"
TISSUE_SAVE_DIR="$(dirname "${WSI_DIR}")/tissue_feats_extraction"

NUCLEI_FEATS_EXTRACTION="$(dirname "${WSI_DIR}")/nuclei_feats_extraction"
TISSUE_FEATS_EXTRACTION="$(dirname "${WSI_DIR}")/tissue_feats_extraction"
POST_SAVE_DIR="$(dirname "${WSI_DIR}")/aggregation"

DATASET_FEATURE_MATRIX="$(dirname "${WSI_DIR}")/aggregation/dataset_feature_matrix.csv"
CROSS_SAVE_DIR="$(dirname "${WSI_DIR}")/cross_validation"

SURVIVAL_SAVE_DIR="$(dirname "${WSI_DIR}")/KM_curve_survival_analysis"

DEG_SAVE_DIR="$(dirname "${WSI_DIR}")/Degs"

ENRICH_SAVE_DIR="$(dirname "${WSI_DIR}")/FEA_GSEA"

# 1: Feature extraction (extract_wsi_feats_v4.py)
echo "[1/6] Feature extraction..."
python ../example/extract_wsi_feats_v4.py \
     --wsi_dir "$WSI_DIR" \
     --nuclei_dir "$NUCLEI_DIR" \
     --tumor_dir "$TUMOR_DIR" \
     --nuclei_save_dir "$NUCLEI_SAVE_DIR" \
     --tissue_roi_dir "$TISSUE_ROI_DIR" \
     --tissue_save_dir "$TISSUE_SAVE_DIR" \


# 2: Postprocessing (postprocessing.py)
echo "[2/6] Postprocessing..."
python ../example/postprocessing.py \
     --nuclei_feats_extraction "$NUCLEI_FEATS_EXTRACTION" \
     --tissue_feats_extraction "$TISSUE_FEATS_EXTRACTION" \
     --label_path "$LABEL_PATH" \
     --post_save_dir "$POST_SAVE_DIR" \


# 3: Multi-fold cross validation (Btrain_test_cross_validationV3.py)
echo "[3/6] Multi-fold cross validation....."
python ../pathomics/model_constructor/Btrain_test_cross_validationV3.py \
     --dataset_feature_matrix "$DATASET_FEATURE_MATRIX" \
     --survival_info_dir "$LABEL_PATH" \
     --cross_save_dir "$CROSS_SAVE_DIR" \


# 4: Kaplan-Meier curve survival analysis (survival_analysis_plotterV2.py)
echo "[4/6] Kaplan-Meier curve survival analysis......"
python ../pathomics/model_constructor/survival_analysis_plotterV2.py \
     --cross_validation "$CROSS_SAVE_DIR" \
     --n_workers 25 \
     --survival_save_dir "$SURVIVAL_SAVE_DIR" \


# 5: Differential expression gene analysis (differential_expression_analyzerV2.py)
echo "[5/6] Differential expression gene screening......."
python ../pathomics/gene_path_analyzer/differential_expression_analyzerV2.py \
     --dataset_feature_matrix "$DATASET_FEATURE_MATRIX" \
     --bulk_rna "$BULK_RNA" \
     --deg_save_dir "$DEG_SAVE_DIR" \
     --cross_validation "$CROSS_SAVE_DIR" \
     --top_signif 2000 \


# 6: Functional enrichment analysis and gene set enrichment analysis (gene_enrichment_analyzerv3.py)
echo "[6/6] FEA and GSEA running......."
python ../pathomics/gene_path_analyzer/gene_enrichment_analyzerv3.py \
     --deg_res "$DEG_SAVE_DIR" \
     --source "GO:All" \
     --clinical_information "$LABEL_PATH" \
     --clinical_col "survival_time" \
     --enrich_save_dir "$ENRICH_SAVE_DIR" \
     --proxy_port "$PROXY_PORT" \
     
echo
echo "✅ The entire process has been executed successfully."
echo "[INFO] End time: $(date)"


