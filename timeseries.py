import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# USER SETTINGS
# =========================
BASE_DIR = Path(
    r"E:\20260206\00 KONKUK\02 Papers\01 SCIE\28th Pests (Timeseries)\python"
)

# KMA 결과 폴더
KMA_DIR = BASE_DIR / "kma_pests_outputs" / "by_event"
KMA_PATTERN = "event_*.csv"

# SMAP 결과 폴더
SMAP_DIR = BASE_DIR / "output_smap_l4_surface_temp_sm_100d_to_5d_1perday"
SMAP_PATTERN = "event_*_SMAP_L4_*surface_temp_sm.csv"

# 그림 저장 폴더
OUT_DIR = BASE_DIR / "plots_timeseries_pests"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# helpers
# =========================
def infer_event_id_from_filename(p: Path) -> int:
    m = re.search(r"event_(\d+)", p.name)
    if not m:
        raise ValueError(f"이벤트 id 파싱 실패: {p.name}")
    return int(m.group(1))


def to_kst_naive(ts) -> pd.Timestamp:
    """
    tz-aware면 KST로 변환 후 tz 제거
    tz-naive면 그대로 반환
    """
    ts = pd.Timestamp(ts)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Seoul").tz_localize(None)
    return ts


def load_kma_event_csv(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"date", "ta", "hm", "rn_day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}에 필요한 KMA 컬럼이 없습니다. 누락: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.floor("D")

    for c in ["ta", "hm", "rn_day"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "event_date" in df.columns:
        event_time = pd.to_datetime(df["event_date"].iloc[0], errors="coerce")
    else:
        event_time = pd.NaT

    event_time = to_kst_naive(event_time)

    eid = int(df["event_id"].iloc[0]) if "event_id" in df.columns else infer_event_id_from_filename(path)
    return eid, event_time, df


def load_smap_event_csv(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"datetime_kst", "surface_temp", "sm_surface"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}에 필요한 SMAP 컬럼이 없습니다. 누락: {sorted(missing)}")

    df["datetime_kst"] = pd.to_datetime(df["datetime_kst"], errors="coerce")
    df = df.dropna(subset=["datetime_kst"]).sort_values("datetime_kst").reset_index(drop=True)

    df["date"] = df["datetime_kst"].apply(to_kst_naive).dt.floor("D")

    df["surface_temp"] = pd.to_numeric(df["surface_temp"], errors="coerce")
    df["surface_temp_C"] = df["surface_temp"] - 273.15
    df["sm_surface"] = pd.to_numeric(df["sm_surface"], errors="coerce")

    if "event_time_kst" in df.columns:
        event_time = pd.to_datetime(df["event_time_kst"].iloc[0], errors="coerce")
    else:
        event_time = pd.NaT

    event_time = to_kst_naive(event_time)

    eid = int(df["event_id"].iloc[0]) if "event_id" in df.columns else infer_event_id_from_filename(path)

    df = df[["date", "surface_temp_C", "sm_surface"]].drop_duplicates(subset=["date"])
    return eid, event_time, df


def build_event_file_maps():
    kma_files = sorted(KMA_DIR.glob(KMA_PATTERN))
    smap_files = sorted(SMAP_DIR.glob(SMAP_PATTERN))

    if not kma_files:
        raise FileNotFoundError(f"KMA 파일을 찾지 못했습니다:\n  folder={KMA_DIR}\n  pattern={KMA_PATTERN}")
    if not smap_files:
        raise FileNotFoundError(f"SMAP 파일을 찾지 못했습니다:\n  folder={SMAP_DIR}\n  pattern={SMAP_PATTERN}")

    kma_map = {infer_event_id_from_filename(p): p for p in kma_files}
    smap_map = {infer_event_id_from_filename(p): p for p in smap_files}

    common_ids = sorted(set(kma_map) & set(smap_map))
    if not common_ids:
        raise RuntimeError("KMA와 SMAP 사이에 공통 event_id가 없습니다.")

    return common_ids, kma_map, smap_map


def merge_event_data(kma_df: pd.DataFrame, smap_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(kma_df, smap_df, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    df["date_plot"] = df["date"]
    return df


# =========================
# plotting
# =========================
def plot_event_timeseries(eid: int, event_time: pd.Timestamp, df: pd.DataFrame, out_dir: Path):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True, constrained_layout=True
    )

    # -------------------------
    # TOP: air temp + surface temp + humidity
    # left y-axis: temperature
    # right y-axis: humidity
    # -------------------------
    ax_top_r = ax_top.twinx()

    ax_top.plot(
        df["date_plot"],
        df["ta"],
        linewidth=1.8,
        label="Air temperature (KMA)"
    )
    ax_top.plot(
        df["date_plot"],
        df["surface_temp_C"],
        linewidth=1.8,
        label="Surface temperature (SMAP)"
    )

    ax_top_r.plot(
        df["date_plot"],
        df["hm"],
        linewidth=1.6,
        linestyle="--",
        label="Relative humidity (KMA)"
    )

    if pd.notna(event_time):
        ax_top.axvline(event_time, color="red", linestyle="--", linewidth=1.6)
        ax_top_r.axvline(event_time, color="red", linestyle="--", linewidth=1.6)

    ax_top.set_ylabel("Temperature (°C)")
    ax_top_r.set_ylabel("Relative humidity (%)")
    ax_top.set_title(f"Pest event {eid}: meteorological and surface-condition time series")
    ax_top.grid(True, alpha=0.3)

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_top_r.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # -------------------------
    # BOTTOM: surface SM + rainfall
    # left y-axis: soil moisture
    # right y-axis: rainfall bar
    # -------------------------
    ax_bot_r = ax_bot.twinx()

    ax_bot.plot(
        df["date_plot"],
        df["sm_surface"],
        linewidth=1.8,
        label="Surface soil moisture (SMAP)"
    )

    ax_bot_r.bar(
        df["date_plot"],
        df["rn_day"],
        width=0.8,
        alpha=0.5,
        label="Daily rainfall (KMA)"
    )

    if pd.notna(event_time):
        ax_bot.axvline(event_time, color="red", linestyle="--", linewidth=1.6)
        ax_bot_r.axvline(event_time, color="red", linestyle="--", linewidth=1.6)

    ax_bot.set_ylabel("Surface soil moisture (m³/m³)")
    ax_bot_r.set_ylabel("Daily rainfall (mm)")
    ax_bot.set_xlabel("Date (KST)")
    ax_bot.grid(True, alpha=0.3)

    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # x-axis formatting
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax_bot.xaxis.set_major_locator(locator)
    ax_bot.xaxis.set_major_formatter(formatter)
    plt.setp(ax_bot.get_xticklabels(), rotation=30, ha="right")

    out_png = out_dir / f"event_{eid:02d}_pests_timeseries.png"
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


# =========================
# main
# =========================
def main():
    common_ids, kma_map, smap_map = build_event_file_maps()
    print(f"[INFO] 공통 event 수: {len(common_ids)}")

    for eid in common_ids:
        try:
            _, kma_event_time, kma_df = load_kma_event_csv(kma_map[eid])
            _, smap_event_time, smap_df = load_smap_event_csv(smap_map[eid])

            # event time은 KMA 우선, 없으면 SMAP 사용
            event_time = kma_event_time if pd.notna(kma_event_time) else smap_event_time
            if pd.notna(event_time):
                event_time = pd.Timestamp(event_time).floor("D")

            df_merge = merge_event_data(kma_df, smap_df)
            plot_event_timeseries(eid, event_time, df_merge, OUT_DIR)

        except Exception as e:
            print(f"[FAIL] event {eid:02d} -> {type(e).__name__}: {e}")

    print("[DONE] all events plotted.")


if __name__ == "__main__":
    main()