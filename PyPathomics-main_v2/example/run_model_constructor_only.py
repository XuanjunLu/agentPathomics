#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单独测试 Model Constructor（建模与交叉验证）：

输入：
- dataset_feature_matrix.csv（必须含列：wsi_id, label，其余为特征列）
- survival_info.csv / xlsx（必须含列：wsi_id, survival_time, event_status）

输出：
- 指定 out-dir 下生成 cross_validation 的各类表格与图（auc_table.csv、topk_features.csv、ROC_curve.png、KM 曲线等）

用法示例（Windows PowerShell）：

python example/run_model_constructor_only.py `
  --dataset-feature-matrix "P:\AI_PyPathomics\PyPathomics-main\example_folder\patient_xxx\aggregation\dataset_feature_matrix.csv" `
  --survival-info "P:\AI_PyPathomics\PyPathomics-main\example_folder\survival_info.csv" `
  --out-dir "P:\AI_PyPathomics\PyPathomics-main\example_folder\patient_xxx\cross_validation_test" `
  --repeats-num 2 --k-fold 2 --n-workers 2

备注：
- `repeats-num` 默认 100 会非常耗时；只做功能连通性测试建议先设为 1~3。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} 不存在: {p}")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model Constructor only (cross_validation)")
    parser.add_argument(
        "--dataset-feature-matrix",
        required=True,
        help='聚合后的特征矩阵 CSV 路径（必须含列 "wsi_id","label"）',
    )
    parser.add_argument(
        "--survival-info",
        required=True,
        help='生存信息路径（csv/xlsx/xls；必须含列 "wsi_id","survival_time","event_status"）',
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录（将写入 auc_table/topk_features/ROC/KM 等产物）",
    )

    parser.add_argument("--top-feature-num", type=int, default=6, help="每个组合选取 top N 特征（默认 6）")
    parser.add_argument("--k-fold", type=int, default=5, help="K 折交叉验证（默认 5）")
    parser.add_argument(
        "--feature-score-method",
        default="addone",
        choices=["addone", "weighted"],
        help='特征得分累计方式（默认 "addone"）',
    )
    parser.add_argument("--var-thresh", type=float, default=0.0, help="方差阈值过滤（默认 0）")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="相关性阈值过滤（默认 0.9）")
    parser.add_argument("--repeats-num", type=int, default=100, help="重复实验次数（默认 100，耗时很长）")

    # 与 GUI 工具默认保持一致：先跑小组合便于验证
    parser.add_argument(
        "--feats-sel",
        nargs="+",
        default=["Lasso", "XGBoost"],
        help='特征选择方法列表（空格分隔），默认：Lasso XGBoost',
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["QDA", "LDA", "RandomForest"],
        help='分类器列表（空格分隔），默认：QDA LDA RandomForest',
    )
    parser.add_argument("--n-workers", type=int, default=3, help="并行进程数（默认 3）")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 延迟导入，确保 sys.path 生效
    from pathomics.model_constructor.Btrain_test_cross_validationV2 import cross_validation  # noqa: E402

    dataset_feature_matrix = Path(args.dataset_feature_matrix).expanduser().resolve()
    survival_info = Path(args.survival_info).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    _ensure_exists(dataset_feature_matrix, "dataset-feature-matrix")
    _ensure_exists(survival_info, "survival-info")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Model Constructor only ===")
    print(f"- dataset_feature_matrix: {dataset_feature_matrix}")
    print(f"- survival_info:          {survival_info}")
    print(f"- out_dir:               {out_dir}")
    print(f"- feats_sel:             {args.feats_sel}")
    print(f"- classifiers:           {args.classifiers}")
    print(f"- k_fold:                {args.k_fold}")
    print(f"- repeats_num:           {args.repeats_num}")
    print(f"- n_workers:             {args.n_workers}")

    cross_validation(
        dataset_feature_matrix,
        survival_info_dir=survival_info,
        save_dir=out_dir,
        top_feature_num=args.top_feature_num,
        k_fold=args.k_fold,
        feature_score_method=args.feature_score_method,
        var_thresh=args.var_thresh,
        corr_threshold=args.corr_threshold,
        repeats_num=args.repeats_num,
        list_feats_selection_args=args.feats_sel,
        list_classifers_args=args.classifiers,
        n_workers=args.n_workers,
    )

    print("\n完成。关键输出通常包括：")
    print(f"- {out_dir / 'auc_table.csv'}")
    print(f"- {out_dir / 'topk_features.csv'}")
    print(f"- {out_dir / 'ROC_curve.png'}")
    print(f"- {out_dir / 'all_the_best_y_test_pred_full_with_time.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

