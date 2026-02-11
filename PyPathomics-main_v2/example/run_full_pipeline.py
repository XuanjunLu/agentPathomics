#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键全流程（WSI → 特征 → 聚合 → 建模 → DEG → 富集）

说明：
- 该脚本是“串联入口”，尽量复用仓库现有实现，不改动核心算法。
- 你需要提前准备：WSI、分割掩膜、label/survival、bulk RNA、临床信息。

用法示例（Windows PowerShell）：

python example/run_full_pipeline.py `
  --wsi-dir "D:\data\svs" `
  --nuclei-dir "D:\data\nuclei_seg" `
  --tumor-dir "D:\data\tumor_seg" `
  --out-dir "D:\out\pypathomics_run" `
  --label-path "D:\data\label.csv" `
  --survival-info "D:\data\survival_info.csv" `
  --bulk-rna "D:\data\bulk_rna.csv" `
  --clinical-info "D:\data\clinical_information.csv" `
  --clinical-col "T_stage" `
  --mag 40 `
  --patch-size 1024 `
  --n-workers 4 `
  --wsi-num 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 让脚本在任何工作目录下运行都能找到 repo 根路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 注意：example/ 目录无 __init__.py，但在 repo root 加入 sys.path 后可按 namespace 包导入
from example.extract_wsi_feats_v4 import extract_patches, nuclei_feats_extraction, tissue_feats_extraction  # noqa: E402
from example.postprocessing import merge_dataset_feature_matrix  # noqa: E402
from pathomics.model_constructor.Btrain_test_cross_validationV2 import cross_validation  # noqa: E402
from pathomics.gene_path_analyzer.differential_expression_analyzerV2 import DifferentialExpressionAnalyzer  # noqa: E402
from pathomics.gene_path_analyzer.gene_enrichment_analyzerv2 import enrichment_analysis  # noqa: E402


logger = logging.getLogger("run_full_pipeline")


def _p(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} 不存在: {p}")


def _guess_mask_paths(wsi_path: Path, nuclei_dir: Path, tumor_dir: Path, mask_suffix: str) -> Tuple[Path, Path, Path]:
    """
    按 v4 脚本约定：
    - nuclei instance: {wsi_name}_instance.{mask_suffix}
    - nuclei class:    {wsi_name}_class.{mask_suffix}
    - tumor mask:      {wsi_name}.{mask_suffix}
    """
    wsi_name = wsi_path.stem
    nuclei_instance = nuclei_dir / f"{wsi_name}_instance.{mask_suffix}"
    nuclei_class = nuclei_dir / f"{wsi_name}_class.{mask_suffix}"
    tumor = tumor_dir / f"{wsi_name}.{mask_suffix}"
    return nuclei_instance, nuclei_class, tumor


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PyPathomics 全流程一键入口（WSI→特征→聚合→建模→DEG→富集）")
    parser.add_argument("--wsi-dir", required=True, help="WSI 文件目录（svs/tif/tiff 等）")
    parser.add_argument("--nuclei-dir", required=True, help="核分割目录（包含 *_instance.* 与 *_class.*）")
    parser.add_argument("--tumor-dir", required=True, help="组织/肿瘤 ROI 掩膜目录（{wsi_name}.{suffix}）")
    parser.add_argument("--roi-dir", default=None, help="（可选）ROI mask 目录，用于 tissue 特征（{wsi_name}.{suffix}）")
    parser.add_argument("--out-dir", required=True, help="输出目录（会创建多个子目录）")

    parser.add_argument("--mask-suffix", default="png", help="mask 文件后缀（默认 png）")
    parser.add_argument("--mag", type=int, default=40, help="目标倍率（默认 40）")
    parser.add_argument("--patch-size", type=int, default=1024, help="patch 大小（默认 1024）")
    parser.add_argument("--n-workers", type=int, default=3, help="进程数（默认 3）")
    parser.add_argument("--wsi-num", type=int, default=None, help="最多处理多少张 WSI（默认全部）")

    parser.add_argument("--skip-patch-extraction", action="store_true", help="跳过 patch 导出（要求 out-dir 下已有 patches/）")

    parser.add_argument("--feature-type", default="shape",
                        choices=["shape", "texture", "topology", "interplay", "all"],
                        help="核相关特征类型（默认 shape）")
    parser.add_argument("--nuclei-type", type=int, default=None, help="（可选）指定核类别（整数），None=全部")

    # 聚合/建模/转录组相关输入
    parser.add_argument("--label-path", required=True, help="标签文件，必须含列 wsi_id,label（csv/xlsx/xls）")
    parser.add_argument("--survival-info", required=True, help="生存信息，必须含列 wsi_id,survival_time,event_status（csv/xlsx/xls）")
    parser.add_argument("--bulk-rna", required=True, help="bulk RNA 表达矩阵（csv，log2(n+1) 形式）")
    parser.add_argument("--clinical-info", required=True, help="临床信息文件（csv/xlsx/xls）")
    parser.add_argument("--clinical-col", default="T_stage", help="临床列名（默认 T_stage）")
    parser.add_argument("--clinical-category", action="store_true", help="临床列是否为分类变量（默认按连续变量处理）")
    parser.add_argument("--deg-top-str", type=int, default=12, help="wsi_id 前 N 位与 bulkRNA 样本名对齐（默认 12）")
    parser.add_argument("--deg-top-signif", type=int, default=2000, help="保留多少个最显著基因（默认 2000）")
    parser.add_argument("--sources", default="GO:All", help="富集数据库来源（默认 GO:All）")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    wsi_dir = _p(args.wsi_dir)
    nuclei_dir = _p(args.nuclei_dir)
    tumor_dir = _p(args.tumor_dir)
    roi_dir = _p(args.roi_dir) if args.roi_dir else None
    out_dir = _p(args.out_dir)

    label_path = _p(args.label_path)
    survival_info = _p(args.survival_info)
    bulk_rna = _p(args.bulk_rna)
    clinical_info = _p(args.clinical_info)

    _ensure_exists(wsi_dir, "wsi-dir")
    _ensure_exists(nuclei_dir, "nuclei-dir")
    _ensure_exists(tumor_dir, "tumor-dir")
    if roi_dir is not None:
        _ensure_exists(roi_dir, "roi-dir")
    _ensure_exists(label_path, "label-path")
    _ensure_exists(survival_info, "survival-info")
    _ensure_exists(bulk_rna, "bulk-rna")
    _ensure_exists(clinical_info, "clinical-info")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一输出子目录（后续多个步骤都会用到）
    nuclei_feats_out = out_dir / "nuclei_feats_extraction"
    tissue_feats_out = out_dir / "tissue_feats_extraction"
    nuclei_feats_out.mkdir(parents=True, exist_ok=True)
    tissue_feats_out.mkdir(parents=True, exist_ok=True)

    # 1) Patch 导出（WSI → patches）
    wsi_paths = sorted([p for p in wsi_dir.glob("*.*") if p.is_file()])
    if args.wsi_num:
        wsi_paths = wsi_paths[: args.wsi_num]
    if not wsi_paths:
        raise FileNotFoundError(f"wsi-dir 下未找到任何文件: {wsi_dir}")

    if not args.skip_patch_extraction:
        logger.info("步骤1/5：WSI patch 导出（可能耗时较长）")
        for wsi_path in wsi_paths:
            nuclei_instance, nuclei_class, tumor = _guess_mask_paths(wsi_path, nuclei_dir, tumor_dir, args.mask_suffix)
            _ensure_exists(nuclei_instance, "nuclei_instance_mask")
            _ensure_exists(nuclei_class, "nuclei_class_mask")
            _ensure_exists(tumor, "tumor_mask")
            extract_patches(
                wsi_path=wsi_path,
                nuclei_instance_path=nuclei_instance,
                nuclei_class_path=nuclei_class,
                tumor_path=tumor,
                # 与 nuclei_feats_extraction() 保持一致：它会在 save_dir/patches/... 读取
                save_dir=nuclei_feats_out,
                mag=args.mag,
                patch_size=args.patch_size,
                n_workers=args.n_workers,
            )
    else:
        logger.info("步骤1/5：跳过 patch 导出（要求 nuclei_feats_extraction/ 下已存在 patches/）")

    # 2) 特征提取（patch → WSI）
    logger.info("步骤2/5：特征提取（nuclei-level & image-level）")
    nuclei_feats_extraction(
        wsi_dir=str(wsi_dir),
        nuclei_dir=str(nuclei_dir),
        tumor_dir=str(tumor_dir),
        save_dir=str(nuclei_feats_out),
        mask_suffix=args.mask_suffix,
        wsi_num=args.wsi_num,
        mag=args.mag,
        n_workers=args.n_workers,
        feature_type=args.feature_type,
        nuclei_type=args.nuclei_type,
    )

    if roi_dir is not None:
        tissue_feats_extraction(
            wsi_dir=str(wsi_dir),
            roi_dir=str(roi_dir),
            save_dir=str(tissue_feats_out),
            mask_suffix=args.mask_suffix,
            wsi_num=args.wsi_num or 200,
            mag=args.mag,
            n_workers=args.n_workers,
        )
    else:
        logger.info("未提供 roi-dir：跳过 tissue 特征提取")

    # 3) 聚合（WSI → dataset_feature_matrix）
    logger.info("步骤3/5：特征聚合生成 dataset_feature_matrix.csv")
    agg_out = out_dir / "aggregation"
    agg_out.mkdir(parents=True, exist_ok=True)

    wsi_image_level_n = nuclei_feats_out / "wsi_image_level"
    wsi_nuclei_level_n = nuclei_feats_out / "wsi_nuclei_level"
    wsi_image_level_t = tissue_feats_out / "wsi_image_level" if roi_dir is not None else Path("__missing__")

    merge_dataset_feature_matrix(
        wsi_image_level_n=wsi_image_level_n,
        wsi_nuclei_level_n=wsi_nuclei_level_n,
        wsi_image_level_t=wsi_image_level_t,
        label_path=label_path,
        save_dir=agg_out,
        raw_data=False,
    )

    dataset_feature_matrix = agg_out / "dataset_feature_matrix.csv"
    _ensure_exists(dataset_feature_matrix, "dataset_feature_matrix.csv")

    # 4) 建模（交叉验证）
    logger.info("步骤4/5：建模与交叉验证")
    cv_out = out_dir / "cross_validation"
    cv_out.mkdir(parents=True, exist_ok=True)

    best = cross_validation(
        dataset_feature_matrix=dataset_feature_matrix,
        survival_info_dir=survival_info,
        save_dir=cv_out,
        top_feature_num=6,
        k_fold=5,
        feature_score_method="addone",
        var_thresh=0,
        corr_threshold=0.9,
        repeats_num=100,
        list_feats_selection_args=["Lasso", "XGBoost"],
        list_classifers_args=["QDA", "LDA", "RandomForest"],
        n_workers=args.n_workers,
    )
    logger.info("最优组合（selection, classifier）：%s", best)

    # 5) DEG + 富集
    logger.info("步骤5/5：DEG + 富集分析/可视化")
    deg_out = out_dir / "degs"
    fea_out = out_dir / "FEA_GSEA"
    deg_out.mkdir(parents=True, exist_ok=True)
    fea_out.mkdir(parents=True, exist_ok=True)

    analyzer = DifferentialExpressionAnalyzer(
        feature_matrix_path=str(dataset_feature_matrix),
        bulk_rna_path=str(bulk_rna),
        save_path=str(deg_out),
        cross_validation_path=str(cv_out),
        top_str=args.deg_top_str,
        top_signif=args.deg_top_signif,
    )
    analyzer.run_analysis()

    enrichment_analysis(
        deg_res=str(deg_out),
        save_dir=str(fea_out),
        sources=args.sources,
        clinical_information=str(clinical_info),
        clinical_col=args.clinical_col,
        category=args.clinical_category,
    )

    logger.info("全流程完成。输出目录：%s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

