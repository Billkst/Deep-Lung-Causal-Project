"""PANCAN-LUAD 数据泄露审计与清洗脚本。

读取 PANCAN 与 LUAD 临床样本 ID，按前 12 位 Patient ID 进行交集匹配，
生成泄露审计报告、黑名单与可用索引。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_patient_id(sample_id: str) -> str:
    sample_id = str(sample_id).strip()
    if not sample_id:
        return ""
    return sample_id[:12]


def _load_first_column_ids(file_path: Path) -> pd.Series:
    df = pd.read_csv(file_path, sep="\t", dtype=str, header=0)
    if df.shape[1] == 0:
        raise ValueError(f"文件为空或无法解析列: {file_path}")
    series = df.iloc[:, 0].astype(str).str.strip()
    series = series[series.notna() & (series != "")]
    return series


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pancan_path = repo_root / "data" / "PANCAN" / "PANCAN_clinical.txt"
    luad_path = repo_root / "data" / "LUAD" / "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix.txt"

    if not pancan_path.exists():
        raise FileNotFoundError(f"未找到 PANCAN 临床文件: {pancan_path}")
    if not luad_path.exists():
        raise FileNotFoundError(f"未找到 LUAD 临床文件: {luad_path}")

    pancan_ids = _load_first_column_ids(pancan_path)
    luad_ids = _load_first_column_ids(luad_path)

    pancan_patient_ids = pancan_ids.map(_normalize_patient_id)
    luad_patient_ids = luad_ids.map(_normalize_patient_id)

    pancan_total = int(pancan_ids.shape[0])
    luad_total = int(luad_ids.shape[0])

    overlap_patient_ids = sorted(set(pancan_patient_ids) & set(luad_patient_ids))
    leakage_mask = pancan_patient_ids.isin(overlap_patient_ids)
    leakage_samples = pancan_ids[leakage_mask]
    leakage_sample_count = int(leakage_samples.shape[0])

    clean_indices = pancan_ids.index[~leakage_mask].to_numpy(dtype=int)

    results_dir = repo_root / "results"
    pancan_dir = repo_root / "data" / "PANCAN"
    results_dir.mkdir(parents=True, exist_ok=True)
    pancan_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / "leakage_audit_report.txt"
    blacklist_path = pancan_dir / "pancan_leakage_blacklist.csv"
    indices_path = pancan_dir / "clean_pancan_indices.npy"

    blacklist_df = pd.DataFrame({"sampleID": leakage_samples.drop_duplicates().sort_values()})
    blacklist_df.to_csv(blacklist_path, index=False)
    np.save(indices_path, clean_indices)

    report_lines = [
        "PANCAN-LUAD 数据泄露审计报告",
        "================================",
        f"PANCAN 总样本数: {pancan_total}",
        f"LUAD 总样本数: {luad_total}",
        f"重叠（泄露）样本数量: {leakage_sample_count}",
        f"重叠 Patient ID 数量: {len(overlap_patient_ids)}",
        "",
        "重叠样本 ID 列表:",
    ]
    report_lines.extend(blacklist_df["sampleID"].tolist())

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    expected_len = pancan_total - leakage_sample_count
    if len(clean_indices) == expected_len:
        print("Verification Passed")
    else:
        raise RuntimeError(
            "Verification Failed: clean indices length mismatch. "
            f"expected={expected_len}, got={len(clean_indices)}"
        )

    print(f"报告已生成: {report_path}")
    print(f"黑名单已生成: {blacklist_path}")
    print(f"索引已生成: {indices_path}")


if __name__ == "__main__":
    main()
