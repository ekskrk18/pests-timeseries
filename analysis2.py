import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = Path(r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\28th Pests (Timeseries)\python")
INPUT_XLSX = BASE_DIR / "pdi_40cases_timeseries_equal_weights.xlsx"

OUT_DIR = BASE_DIR / "persistence_metric_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "persistence_metric_summary.csv"
OUT_BOXPLOT = OUT_DIR / "persistence_boxplot.png"
OUT_HEATMAP = OUT_DIR / "persistence_heatmap.png"

# Persistence definition
THRESHOLD_REF_WINDOW = (-30, 0)   # threshold 계산용
PERSISTENCE_WINDOW = (-14, 0)     # 지속성 계산용
THRESHOLD_QUANTILE = 0.75

# Heatmap style
HEATMAP_CMAP = "YlOrRd"
HEATMAP_PAD_RATIO = 0.08


# =========================================================
# HELPERS
# =========================================================
def parse_pdi_name(colname: str):
    """
    예: PDI_ta_R3_P7 -> temp_case='ta', rain_window=3, persist_window=7
    """
    m = re.match(r"PDI_(ta|sfc)_R(\d+)_P(\d+)", str(colname))
    if not m:
        return None
    return {
        "temp_case": m.group(1),
        "rain_window": int(m.group(2)),
        "persist_window": int(m.group(3)),
    }


def load_event_sheets(xlsx_path: Path):
    book = pd.read_excel(xlsx_path, sheet_name=None)
    event_sheets = {}

    for sheet_name, df in book.items():
        s = str(sheet_name).strip().lower()

        if s in ["log", "summary", "readme", "notes"]:
            continue

        if not re.search(r"event[\s_]*\d+", s):
            continue

        event_sheets[sheet_name] = df.copy()

    return event_sheets


def get_event_id(sheet_name: str, df: pd.DataFrame):
    if "event_id" in df.columns:
        vals = pd.to_numeric(df["event_id"], errors="coerce").dropna()
        if len(vals) > 0:
            return int(vals.iloc[0])

    m = re.search(r"(\d+)", str(sheet_name))
    if m:
        return int(m.group(1))

    raise ValueError(f"event_id를 찾지 못했습니다: {sheet_name}")


def calc_persistence_score(df: pd.DataFrame, pdi_col: str):
    """
    Persistence score:
      - threshold = 75th percentile of PDI in [-30, 0]
      - persistence = number of days in [-14, 0] where PDI > threshold

    Returns:
      persistence_score, threshold
    """
    work = df[["relative_day", pdi_col]].copy()
    work["relative_day"] = pd.to_numeric(work["relative_day"], errors="coerce")
    work[pdi_col] = pd.to_numeric(work[pdi_col], errors="coerce")
    work = work.dropna(subset=["relative_day"]).sort_values("relative_day")

    # threshold reference window
    ref_vals = work.loc[
        (work["relative_day"] >= THRESHOLD_REF_WINDOW[0]) &
        (work["relative_day"] <= THRESHOLD_REF_WINDOW[1]),
        pdi_col
    ].dropna()

    if len(ref_vals) == 0:
        return np.nan, np.nan

    threshold = ref_vals.quantile(THRESHOLD_QUANTILE)

    # persistence window
    target_vals = work.loc[
        (work["relative_day"] >= PERSISTENCE_WINDOW[0]) &
        (work["relative_day"] <= PERSISTENCE_WINDOW[1]),
        pdi_col
    ].dropna()

    if len(target_vals) == 0:
        return np.nan, threshold

    persistence_score = int((target_vals > threshold).sum())
    return persistence_score, threshold


# =========================================================
# MAIN CALCULATION
# =========================================================
def main():
    event_sheets = load_event_sheets(INPUT_XLSX)
    if not event_sheets:
        raise FileNotFoundError("이벤트 시트를 찾지 못했습니다.")

    rows = []

    for sheet_name, df in event_sheets.items():
        event_id = get_event_id(sheet_name, df)

        if "relative_day" not in df.columns:
            raise ValueError(f"{sheet_name}에 relative_day 컬럼이 없습니다.")

        pdi_cols = [c for c in df.columns if str(c).startswith("PDI_")]

        for col in pdi_cols:
            info = parse_pdi_name(col)
            if info is None:
                continue

            persistence_score, threshold = calc_persistence_score(df, col)

            rows.append({
                "event_id": event_id,
                "sheet_name": sheet_name,
                "pdi_case": col,
                "temp_case": info["temp_case"],
                "rain_window": info["rain_window"],
                "persist_window": info["persist_window"],
                "persistence_score": persistence_score,
                "threshold": threshold,
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("계산된 persistence 결과가 없습니다.")

    summary = summary.sort_values(
        ["temp_case", "rain_window", "persist_window", "event_id"]
    ).reset_index(drop=True)

    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {OUT_CSV}")

    # =====================================================
    # BOX PLOT
    # =====================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)

    for ax, temp_case, title in zip(
        axes,
        ["ta", "sfc"],
        ["Air temperature-based", "Surface temperature-based"]
    ):
        sub = summary[summary["temp_case"] == temp_case].copy()

        order = []
        for r in [1, 3, 5, 7]:
            for p in [1, 3, 7, 15, 30]:
                order.append((r, p))

        data = []
        labels = []
        for r, p in order:
            vals = sub.loc[
                (sub["rain_window"] == r) & (sub["persist_window"] == p),
                "persistence_score"
            ].dropna().values
            data.append(vals)
            labels.append(f"R{r}\nP{p}")

        ax.boxplot(data, patch_artist=True, showfliers=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Persistence score")
        ax.set_title(f"{title} | Persistence metric box plot")
        ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(OUT_BOXPLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {OUT_BOXPLOT}")

    # =====================================================
    # HEATMAP
    # =====================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    pivots = {}
    all_heat_values = []

    for temp_case in ["ta", "sfc"]:
        sub = summary[summary["temp_case"] == temp_case].copy()

        pivot = sub.pivot_table(
            index="rain_window",
            columns="persist_window",
            values="persistence_score",
            aggfunc="mean"
        ).reindex(index=[1, 3, 5, 7], columns=[1, 3, 7, 15, 30])

        pivots[temp_case] = pivot
        valid_vals = pivot.values[np.isfinite(pivot.values)]
        all_heat_values.extend(valid_vals)

    all_heat_values = np.array(all_heat_values, dtype=float)

    vmin_raw = np.nanmin(all_heat_values)
    vmax_raw = np.nanmax(all_heat_values)
    vrange = vmax_raw - vmin_raw
    pad = vrange * HEATMAP_PAD_RATIO if vrange > 0 else 0.5

    vmin = vmin_raw - pad
    vmax = vmax_raw + pad

    for ax, temp_case, title in zip(
        axes,
        ["ta", "sfc"],
        ["Air temperature-based", "Surface temperature-based"]
    ):
        pivot = pivots[temp_case]

        im = ax.imshow(
            pivot.values,
            cmap=HEATMAP_CMAP,
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"P{c}" for c in pivot.columns], fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"R{r}" for r in pivot.index], fontsize=11)
        ax.set_title(f"{title} | Mean persistence score", fontsize=14)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white", fontsize=10
                    )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Persistence score", fontsize=12)

    fig.savefig(OUT_HEATMAP, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {OUT_HEATMAP}")

    print("[DONE] Persistence metric analysis complete.")


if __name__ == "__main__":
    main()