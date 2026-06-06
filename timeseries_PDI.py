import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = Path(r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\28th Pests (Timeseries)\python")

KMA_FILE = BASE_DIR / "kma_pests_daily_timeseries.xlsx"
SMAP_FILE = BASE_DIR / "smap_l4_surface_temp_sm_timeseries_1perday.xlsx"

OUT_DIR = BASE_DIR / "pdi_timeseries_outputs_equal_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EXCEL = OUT_DIR / "pdi_40cases_timeseries_equal_weights.xlsx"
OUT_CSV_ALL = OUT_DIR / "pdi_40cases_timeseries_equal_weights_all.csv"
OUT_LOG = OUT_DIR / "pdi_generation_log.csv"

# ---------------------------------------------------------
# F1 temperature suitability settings
# ---------------------------------------------------------
# <= 10°C: 0
# 10-20°C: linearly increase to 1
# 20-28°C: 1
# 28-35°C: linearly decrease to 0
# >= 35°C: 0
T_MIN0 = 10.0
T_MIN1 = 20.0
T_MAX1 = 28.0
T_MAX0 = 35.0

# F3 rain windows
RAIN_WINDOWS = [1, 3, 5, 7]

# F4 persistence windows
PERSIST_WINDOWS = [1, 3, 7, 15, 30]

# Equal weights
W_F1 = 0.25
W_F2 = 0.25
W_F3 = 0.25
W_F4 = 0.25

# 강제로 event 1~10 생성
TARGET_EVENT_IDS = list(range(1, 11))


# =========================================================
# HELPERS
# =========================================================
def minmax_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    vmin = s.min(skipna=True)
    vmax = s.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=s.index)
    return (s - vmin) / (vmax - vmin)


def trapezoid_suitability(temp_c: pd.Series,
                          t_min0=T_MIN0, t_min1=T_MIN1,
                          t_max1=T_MAX1, t_max0=T_MAX0) -> pd.Series:
    x = pd.to_numeric(temp_c, errors="coerce")
    y = pd.Series(np.nan, index=x.index, dtype=float)

    y[x <= t_min0] = 0.0
    y[x >= t_max0] = 0.0

    m1 = (x > t_min0) & (x < t_min1)
    y.loc[m1] = (x.loc[m1] - t_min0) / (t_min1 - t_min0)

    m2 = (x >= t_min1) & (x <= t_max1)
    y.loc[m2] = 1.0

    m3 = (x > t_max1) & (x < t_max0)
    y.loc[m3] = (t_max0 - x.loc[m3]) / (t_max0 - t_max1)

    return y.clip(lower=0.0, upper=1.0)


def infer_event_id_from_name(name: str):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else None


def load_event_sheets(xlsx_path: Path) -> dict:
    return pd.read_excel(xlsx_path, sheet_name=None)


def normalize_within_event(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c + "_norm"] = minmax_01(out[c])
    return out


def ensure_temp_c(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return s
    if s.median(skipna=True) > 100:
        return s - 273.15
    return s


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[c]):
            out[c] = out[c].dt.tz_localize(None)
    return out


def get_sheet_event_id(df: pd.DataFrame, sheet_name: str):
    if "event_id" in df.columns:
        vals = pd.to_numeric(df["event_id"], errors="coerce").dropna()
        if len(vals) > 0:
            return int(vals.iloc[0])
    return infer_event_id_from_name(sheet_name)


# =========================================================
# LOAD + INDEX SHEETS BY EVENT_ID
# =========================================================
def build_event_sheet_map(book: dict, source_name: str) -> dict:
    event_map = {}
    for sheet_name, df in book.items():
        eid = get_sheet_event_id(df, sheet_name)
        if eid is None:
            print(f"[WARN] {source_name}: sheet '{sheet_name}'에서 event_id를 찾지 못해 건너뜀")
            continue

        if eid in event_map:
            print(f"[WARN] {source_name}: event_id={eid}가 중복입니다. 기존 sheet='{event_map[eid][0]}', 새 sheet='{sheet_name}'. 새 sheet로 덮어씀")
        event_map[eid] = (sheet_name, df.copy())

    return event_map


def prepare_kma_df(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError(f"KMA sheet '{sheet_name}'에 date 컬럼이 없습니다.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).copy()

    if "event_id" not in df.columns:
        df["event_id"] = infer_event_id_from_name(sheet_name)

    for c in ["ta", "hm", "rn_day"]:
        if c not in df.columns:
            raise ValueError(f"KMA sheet '{sheet_name}'에 '{c}' 컬럼이 없습니다.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "relative_day" not in df.columns:
        if "event_date" in df.columns and df["event_date"].notna().any():
            event_date = pd.to_datetime(df["event_date"].iloc[0], errors="coerce")
            df["relative_day"] = (df["date"] - event_date.floor("D")).dt.days
        else:
            raise ValueError(f"KMA sheet '{sheet_name}'에 relative_day 또는 event_date가 없습니다.")

    keep = [c for c in ["event_id", "date", "relative_day", "event_date", "ta", "hm", "rn_day"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["event_id", "date"]).sort_values("date").reset_index(drop=True)


def prepare_smap_df(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        if "datetime_kst" in df.columns:
            df["date"] = pd.to_datetime(df["datetime_kst"], errors="coerce").dt.floor("D")
        else:
            raise ValueError(f"SMAP sheet '{sheet_name}'에 date 또는 datetime_kst 컬럼이 없습니다.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).copy()

    if "event_id" not in df.columns:
        df["event_id"] = infer_event_id_from_name(sheet_name)

    if "surface_temp_C" not in df.columns:
        if "surface_temp" not in df.columns:
            raise ValueError(f"SMAP sheet '{sheet_name}'에 surface_temp 또는 surface_temp_C가 없습니다.")
        df["surface_temp_C"] = ensure_temp_c(df, "surface_temp")
    else:
        df["surface_temp_C"] = pd.to_numeric(df["surface_temp_C"], errors="coerce")

    if "sm_surface" not in df.columns:
        raise ValueError(f"SMAP sheet '{sheet_name}'에 sm_surface 컬럼이 없습니다.")

    df["sm_surface"] = pd.to_numeric(df["sm_surface"], errors="coerce")

    keep = [c for c in ["event_id", "date", "sm_surface", "surface_temp_C"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["event_id", "date"]).sort_values("date").reset_index(drop=True)


def merge_one_event(kma_df: pd.DataFrame, smap_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        kma_df,
        smap_df,
        on=["event_id", "date"],
        how="outer"
    ).sort_values("date").reset_index(drop=True)

    if "relative_day" not in df.columns or df["relative_day"].isna().all():
        if "event_date" in df.columns and df["event_date"].notna().any():
            event_date = pd.to_datetime(df["event_date"].iloc[0], errors="coerce")
            df["relative_day"] = (df["date"] - event_date.floor("D")).dt.days
        else:
            zero_date = df["date"].median()
            df["relative_day"] = (df["date"] - zero_date.floor("D")).dt.days

    return df


# =========================================================
# FACTORS + PDI
# =========================================================
def compute_factors_and_pdi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["ta", "hm", "rn_day", "sm_surface", "surface_temp_C"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # F1
    out["F1_ta"] = trapezoid_suitability(out["ta"])
    out["F1_sfc"] = trapezoid_suitability(out["surface_temp_C"])

    # F2
    out = normalize_within_event(out, ["hm", "sm_surface", "rn_day"])
    out["F2"] = 0.5 * out["hm_norm"] + 0.5 * out["sm_surface_norm"]

    # F1+F2 base
    out["F12_ta_base"] = 0.5 * out["F1_ta"] + 0.5 * out["F2"]
    out["F12_sfc_base"] = 0.5 * out["F1_sfc"] + 0.5 * out["F2"]

    # F3
    for rw in RAIN_WINDOWS:
        raw_name = f"F3_rainsum_{rw}d_raw"
        score_name = f"F3_rain_{rw}d"
        out[raw_name] = out["rn_day"].rolling(rw, min_periods=1).sum()
        out[score_name] = minmax_01(out[raw_name])

    # F4
    for pw in PERSIST_WINDOWS:
        out[f"F4_ta_persist_{pw}d"] = out["F12_ta_base"].rolling(pw, min_periods=1).mean()
        out[f"F4_sfc_persist_{pw}d"] = out["F12_sfc_base"].rolling(pw, min_periods=1).mean()

    # Equal-weight PDI: 2 * 4 * 5 = 40
    for temp_case in ["ta", "sfc"]:
        f1_col = "F1_ta" if temp_case == "ta" else "F1_sfc"

        for rw in RAIN_WINDOWS:
            f3_col = f"F3_rain_{rw}d"

            for pw in PERSIST_WINDOWS:
                f4_col = f"F4_{temp_case}_persist_{pw}d"
                pdi_col = f"PDI_{temp_case}_R{rw}_P{pw}"

                out[pdi_col] = (
                    W_F1 * out[f1_col] +
                    W_F2 * out["F2"] +
                    W_F3 * out[f3_col] +
                    W_F4 * out[f4_col]
                )

    return out


# =========================================================
# PLOTTING
# =========================================================
def plot_event_pdi(event_id: int, df: pd.DataFrame, out_dir: Path):
    df = df.sort_values("date").reset_index(drop=True)

    event_date = None
    if "event_date" in df.columns and df["event_date"].notna().any():
        event_date = pd.to_datetime(df["event_date"].iloc[0], errors="coerce")
    if pd.isna(event_date):
        tmp = df.loc[df["relative_day"] == 0, "date"]
        if len(tmp) > 0:
            event_date = pd.to_datetime(tmp.iloc[0], errors="coerce")

    for temp_case, title_name in [("ta", "Air temperature-based"), ("sfc", "Surface temperature-based")]:
        cols = [c for c in df.columns if c.startswith(f"PDI_{temp_case}_")]
        cols = sorted(cols)

        fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)

        for c in cols:
            ax.plot(df["date"], df[c], linewidth=1.1, label=c)

        if event_date is not None and not pd.isna(event_date):
            ax.axvline(event_date, color="red", linestyle="--", linewidth=1.6, label="Event date")

        ax.set_title(f"Event {event_id} | {title_name} | 20 PDI cases (equal weights)")
        ax.set_ylabel("PDI")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.legend(
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=True,
            framealpha=0.9  # 약간 투명하게 (겹침 완화)
        )

        out_png = out_dir / f"event_{event_id:02d}_{temp_case}_20cases_equal_weights.png"
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("[INFO] loading workbooks...")
    kma_book = load_event_sheets(KMA_FILE)
    smap_book = load_event_sheets(SMAP_FILE)

    kma_event_map = build_event_sheet_map(kma_book, "KMA")
    smap_event_map = build_event_sheet_map(smap_book, "SMAP")

    print(f"[INFO] KMA event ids : {sorted(kma_event_map.keys())}")
    print(f"[INFO] SMAP event ids: {sorted(smap_event_map.keys())}")

    out_book = {}
    all_rows = []
    logs = []

    for event_id in TARGET_EVENT_IDS:
        print(f"\n[INFO] processing event {event_id}...")

        if event_id not in kma_event_map:
            msg = f"KMA workbook에 event {event_id}가 없습니다."
            print(f"[WARN] {msg}")
            logs.append({"event_id": event_id, "status": "missing_kma", "message": msg})
            continue

        if event_id not in smap_event_map:
            msg = f"SMAP workbook에 event {event_id}가 없습니다."
            print(f"[WARN] {msg}")
            logs.append({"event_id": event_id, "status": "missing_smap", "message": msg})
            continue

        kma_sheet, kma_raw = kma_event_map[event_id]
        smap_sheet, smap_raw = smap_event_map[event_id]

        try:
            kma_df = prepare_kma_df(kma_raw, kma_sheet)
            smap_df = prepare_smap_df(smap_raw, smap_sheet)
            merged_df = merge_one_event(kma_df, smap_df)

            df_out = compute_factors_and_pdi(merged_df)
            out_book[f"event_{event_id:02d}"] = make_excel_safe(df_out)
            all_rows.append(df_out)

            plot_event_pdi(event_id, df_out, PLOT_DIR)

            logs.append({
                "event_id": event_id,
                "status": "success",
                "message": f"KMA='{kma_sheet}', SMAP='{smap_sheet}', rows={len(df_out)}"
            })

        except Exception as e:
            print(f"[FAIL] event {event_id}: {type(e).__name__}: {e}")
            logs.append({
                "event_id": event_id,
                "status": "failed",
                "message": f"{type(e).__name__}: {e}"
            })

    if not out_book:
        raise RuntimeError("생성된 event 결과가 없습니다.")

    print("\n[INFO] writing Excel...")
    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        for sheet, sdf in out_book.items():
            sdf.to_excel(writer, sheet_name=sheet[:31], index=False)

        pd.DataFrame(logs).to_excel(writer, sheet_name="log", index=False)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(OUT_CSV_ALL, index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

    print(f"[DONE] Excel: {OUT_EXCEL}")
    print(f"[DONE] CSV  : {OUT_CSV_ALL}")
    print(f"[DONE] Log  : {OUT_LOG}")
    print(f"[DONE] Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()