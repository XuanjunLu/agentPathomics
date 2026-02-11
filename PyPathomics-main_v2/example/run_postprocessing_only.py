#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单独测试 postprocessing（把提取后的特征目录聚合成 dataset_feature_matrix.csv）。

输入（目录/文件）：
- wsi_image_level_n：nuclei 图像级特征目录（CSV 集合）
- wsi_nuclei_level_n：nuclei 单核特征目录（CSV 集合，含 tumor_status / nuclei_class 等列）
- wsi_image_level_t：tissue 图像级特征目录（CSV 集合）
- label_path：标签文件（csv/xlsx/xls），必须含列 wsi_id,label

输出：
- save_dir/dataset_feature_matrix.csv

用法示例（PowerShell）：

python example/run_postprocessing_only.py `
  --wsi-image-level-n "P:\\AI_PyPathomics\\PyPathomics-main\\example_folder\\patient_xxx\\nuclei_feats_extraction\\wsi_image_level" `
  --wsi-nuclei-level-n "P:\\AI_PyPathomics\\PyPathomics-main\\example_folder\\patient_xxx\\nuclei_feats_extraction\\wsi_nuclei_level" `
  --wsi-image-level-t "P:\\AI_PyPathomics\\PyPathomics-main\\example_folder\\patient_xxx\\tissue_feats_extraction\\wsi_image_level" `
  --label-path "P:\\AI_PyPathomics\\PyPathomics-main\\example_folder\\survival_info.csv" `
  --save-dir "P:\\AI_PyPathomics\\PyPathomics-main\\example_folder\\patient_xxx\\aggregation_rebuild"
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
    parser = argparse.ArgumentParser(description="Run postprocessing only (merge_dataset_feature_matrix)")
    parser.add_argument("--wsi-image-level-n", required=True, help="nuclei 图像级特征目录（wsi_image_level）")
    parser.add_argument("--wsi-nuclei-level-n", required=True, help="nuclei 单核特征目录（wsi_nuclei_level）")
    parser.add_argument("--wsi-image-level-t", required=True, help="tissue 图像级特征目录（wsi_image_level）")
    parser.add_argument("--label-path", required=True, help='标签文件（必须含列 "wsi_id","label"）')
    parser.add_argument("--save-dir", required=True, help="输出目录（写入 dataset_feature_matrix.csv）")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from example.postprocessing import merge_dataset_feature_matrix  # noqa: E402

    wsi_image_level_n = Path(args.wsi_image_level_n).expanduser().resolve()
    wsi_nuclei_level_n = Path(args.wsi_nuclei_level_n).expanduser().resolve()
    wsi_image_level_t = Path(args.wsi_image_level_t).expanduser().resolve()
    label_path = Path(args.label_path).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()

    _ensure_exists(wsi_image_level_n, "wsi-image-level-n")
    _ensure_exists(wsi_nuclei_level_n, "wsi-nuclei-level-n")
    _ensure_exists(wsi_image_level_t, "wsi-image-level-t")
    _ensure_exists(label_path, "label-path")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== postprocessing only ===")
    print(f"- wsi_image_level_n: {wsi_image_level_n}")
    print(f"- wsi_nuclei_level_n: {wsi_nuclei_level_n}")
    print(f"- wsi_image_level_t: {wsi_image_level_t}")
    print(f"- label_path: {label_path}")
    print(f"- save_dir: {save_dir}")

    merge_dataset_feature_matrix(
        wsi_image_level_n=wsi_image_level_n,
        wsi_nuclei_level_n=wsi_nuclei_level_n,
        wsi_image_level_t=wsi_image_level_t,
        label_path=label_path,
        save_dir=save_dir,
        raw_data=False,
    )

    out_csv = save_dir / "dataset_feature_matrix.csv"
    print(f"\n完成：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

