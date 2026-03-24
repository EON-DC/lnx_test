from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_filename(name: str, max_len: int = 80) -> str:
    """Make a filesystem-safe filename stem."""
    name = str(name)
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)  # windows forbidden
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:max_len] if len(name) > max_len else name


def _md_table(df: pd.DataFrame) -> str:
    """Convert to markdown table. Fallback if tabulate isn't installed."""
    try:
        return df.to_markdown()
    except Exception:
        # minimal fallback
        lines = []
        cols = list(df.columns)
        lines.append("| " + " | ".join(map(str, cols)) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(map(lambda x: str(x), row.values)) + " |")
        return "\n".join(lines)


def write_col_section(
    df: pd.DataFrame,
    col: str,
    *,
    top_n: int = 20,
    bins: int = 30,
    save_plots: bool = True,
    plots_dir: str = "plots",
    dropna_for_plots: bool = True,
) -> Tuple[str, List[str]]:
    """
    Return (markdown_section, saved_image_paths_relative_to_md).
    """
    if col not in df.columns:
        return f"## Column: `{col}`\n\n- ❌ Column not found.\n\n", []

    s = df[col]
    n = len(s)
    null_cnt = int(s.isna().sum())
    null_ratio = (null_cnt / n) if n else np.nan
    nunique = int(s.nunique(dropna=True))
    is_unique = bool(s.is_unique)
    dtype = str(s.dtype)

    # ---- type detection ----
    is_bool = pd.api.types.is_bool_dtype(s)
    is_numeric = pd.api.types.is_numeric_dtype(s) and not is_bool

    md: List[str] = []
    saved_imgs: List[str] = []

    md.append(f"## Column: `{col}`")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| dtype | `{dtype}` |")
    md.append(f"| rows | {n:,} |")
    md.append(f"| nunique (dropna) | {nunique:,} |")
    md.append(f"| is_unique | {is_unique} |")
    md.append(f"| null_count | {null_cnt:,} |")
    md.append(f"| null_ratio | {null_ratio:.4%} |")
    md.append("")

    # value_counts / describe
    if not is_unique:
        if is_numeric:
            desc = s.describe()
            md.append("### Describe (numeric)")
            md.append(_md_table(desc.to_frame("value")))
            md.append("")
        else:
            vc = s.value_counts(dropna=False).head(top_n)
            md.append(f"### Value counts (top {top_n}, include NaN)")
            md.append(_md_table(vc.to_frame("count")))
            md.append("")
    else:
        md.append("> ✅ All values are unique (frequency summary skipped).")
        md.append("")

    if save_plots:
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(col)

        sp = s.dropna() if dropna_for_plots else s

        # 1) Boolean -> bar chart (NOT hist/box)
        if is_bool:
            vc = s.value_counts(dropna=False)
            bar_path = Path(plots_dir) / f"{safe}__bool.png"
            plt.figure(figsize=(6, 4))
            plt.bar(vc.index.astype(str), vc.values)
            plt.title(f"Boolean counts - {col}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()
            saved_imgs.append(str(bar_path))

            md.append("### Plot")
            md.append(f"![]({bar_path.as_posix()})")
            md.append("")

        # 2) Numeric (except bool) -> hist + box
        elif is_numeric:
            hist_path = Path(plots_dir) / f"{safe}__hist.png"
            plt.figure(figsize=(10, 4))
            plt.hist(sp.values, bins=bins)
            plt.title(f"Histogram - {col}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=150)
            plt.close()
            saved_imgs.append(str(hist_path))

            box_path = Path(plots_dir) / f"{safe}__box.png"
            plt.figure(figsize=(10, 2.6))
            plt.boxplot(sp.values, vert=False)
            plt.title(f"Boxplot - {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(box_path, dpi=150)
            plt.close()
            saved_imgs.append(str(box_path))

            md.append("### Plots")
            md.append(f"![]({hist_path.as_posix()})")
            md.append(f"![]({box_path.as_posix()})")
            md.append("")

        # 3) Non-numeric -> top categories bar
        else:
            vc = s.value_counts(dropna=False).head(top_n)
            bar_path = Path(plots_dir) / f"{safe}__top{top_n}.png"
            plt.figure(figsize=(10, 4))
            plt.bar(vc.index.astype(str), vc.values)
            plt.title(f"Top {top_n} categories - {col}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=150)
            plt.close()
            saved_imgs.append(str(bar_path))

            md.append("### Plot")
            md.append(f"![]({bar_path.as_posix()})")
            md.append("")

    return "\n".join(md) + "\n", saved_imgs


def write_summary_md(
    df: pd.DataFrame,
    *,
    out_md_path: str = "summary.md",
    only_non_unique: bool = True,
    sort_by: str = "null_ratio",  # "null_ratio" | "nunique" | "col"
    top_n: int = 20,
    bins: int = 30,
    save_plots: bool = True,
    plots_dir: str = "plots",
) -> str:
    """
    Create a markdown report file and return the output path.
    """
    # dataset-level overview table
    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        null_cnt = int(s.isna().sum())
        null_ratio = (null_cnt / n) if n else np.nan
        rows.append(
            {
                "col": col,
                "dtype": str(s.dtype),
                "rows": n,
                "nunique(dropna)": int(s.nunique(dropna=True)),
                "is_unique": bool(s.is_unique),
                "null_count": null_cnt,
                "null_ratio": float(null_ratio),
            }
        )
    overview = pd.DataFrame(rows)

    if only_non_unique:
        target = overview[overview["is_unique"] == False].copy()
    else:
        target = overview.copy()

    if sort_by == "null_ratio":
        target = target.sort_values("null_ratio", ascending=False)
    elif sort_by == "nunique":
        target = target.sort_values("nunique(dropna)", ascending=False)
    else:
        target = target.sort_values("col", ascending=True)

    # write file
    out_path = Path(out_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_parts: List[str] = []
    md_parts.append("# EDA Summary")
    md_parts.append("")
    md_parts.append(f"- rows: **{len(df):,}**")
    md_parts.append(f"- cols: **{df.shape[1]:,}**")
    md_parts.append(
        f"- report scope: **{'non-unique columns only' if only_non_unique else 'all columns'}**"
    )
    md_parts.append("")
    md_parts.append("## Overview")
    md_parts.append(
        _md_table(
            target.assign(
                null_ratio=lambda x: (x["null_ratio"] * 100).round(2).astype(str) + "%"
            ).drop(columns=["null_ratio"])
        )
    )
    md_parts.append("")

    md_parts.append("---\n")

    # per-column sections
    for col in target["col"].tolist():
        section, _ = write_col_section(
            df,
            col,
            top_n=top_n,
            bins=bins,
            save_plots=save_plots,
            plots_dir=plots_dir,
        )
        md_parts.append(section)
        md_parts.append("---\n")

    out_path.write_text("\n".join(md_parts), encoding="utf-8")
    return str(out_path)
