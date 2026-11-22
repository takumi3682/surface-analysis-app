from datetime import datetime, time, timedelta, timezone
import io
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature


if Path("/Users").exists():
    for d in [Path.cwd(), Path.cwd().parent, Path("/Users/takumi/Projects/meteo_lib")]:
        if d.exists() and str(d.resolve()) not in sys.path:
            sys.path.append(str(d.resolve()))

from meteo_lib.obs_utils import get_synop_metar_combined
from meteo_lib.plot_utils import (
    plot_amedas_barbs_and_obs,
    plot_amedas_temp_shade,
    plot_msm_isobars,
    plot_synop_metar_station,
    setup_map_axis,
)
from meteo_lib.surface_utils import (
    REGION_EN,
    REGION_EXTENTS,
    get_msm_surface,
    load_amedas_parquet,
)

st.set_page_config(
    page_title="地上天気図解析 (Surface Analysis)",
    layout="wide",
)

IS_LOCAL = Path("/Users").exists()
DEFAULT_AMEDAS_DIR = Path("/Users/takumi/Projects/analysis-Meteo/data/amedas_parquet")

CACHE_SURF_DIR = (
    "/Users/takumi/Projects/cache/surface" if IS_LOCAL else "/tmp/meteo_cache/surface"
)
RAW_GRIB_DIR = (
    "/Users/takumi/Projects/raw_grib" if IS_LOCAL else "/tmp/meteo_raw_grib"
)


@st.cache_data(ttl=3600, show_spinner="MSM 地上データを取得・解析中...")
def cached_get_msm(dt_str: str, extent_tuple: tuple):
    ds = get_msm_surface(
        dt_str,
        extent=extent_tuple,
        cache_dir=CACHE_SURF_DIR,
        raw_dir=RAW_GRIB_DIR,
    )
    if not IS_LOCAL and Path(RAW_GRIB_DIR).exists():
        for f in Path(RAW_GRIB_DIR).glob(f"*{dt_str}*"):
            try:
                f.unlink()
            except OSError:
                pass
    return ds.load() if ds is not None else None


@st.cache_data(ttl=1800, show_spinner="SYNOP / METAR データを取得中...")
def cached_get_obs(target_iso: str, extent_tuple: tuple, min_dist: float):
    dt_utc = pd.to_datetime(target_iso, utc=True)
    return get_synop_metar_combined(
        dt_utc, list(extent_tuple), min_distance_deg=min_dist
    )


@st.cache_data(ttl=3600, show_spinner="アメダスデータを読み込み中...")
def cached_load_amedas(base_dir_str: str, target_iso: str, extent_tuple: tuple):
    dt_utc = pd.to_datetime(target_iso, utc=True)
    return load_amedas_parquet(base_dir_str, dt_utc, list(extent_tuple))


def create_surface_figure(
    target_dt: pd.Timestamp,
    region: str,
    amedas_dir: Path,
    show_isobars: bool,
    show_amedas_temp_shade: bool,
    show_amedas_obs: bool,
    show_amedas_wind: bool,
    show_synop_metar: bool,
    show_station_name: bool,
    msm_sigma: float,
    temp_sigma: float,
    temp_vmin: float,
    temp_vmax: float,
    temp_cmap: str,
):
    dt_str = target_dt.strftime("%Y%m%d%H")
    target_iso = target_dt.isoformat()
    extent = REGION_EXTENTS.get(region, REGION_EXTENTS["関東"])
    extent_tuple = tuple(extent)

    is_regional = (extent[1] - extent[0]) < 6.0
    station_min_dist = 0.20 if is_regional else 0.8

#    fig, ax, proj = setup_map_axis(extent, figsize=(11, 10), dpi=180)


    # create_surface_figure 内の setup_map_axis の代わり
    fig = plt.figure(figsize=(10, 9), dpi=150 if IS_LOCAL else 120)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)

    coast_res = "10m" if IS_LOCAL else "50m"
    ax.add_feature(
        cfeature.OCEAN.with_scale(coast_res),
        facecolor="#ffffff",
        edgecolor="none",
        zorder=2,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale(coast_res),
        edgecolor="#333333",
        linewidth=0.9,
        zorder=3,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(coast_res),
        linestyle=":",
        edgecolor="#666666",
        zorder=3,
    )
    gl = ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        alpha=0.3,
        zorder=3,
    )
    gl.top_labels = False
    gl.right_labels = False

    has_amedas = amedas_dir.exists()
    # アメダスデータ取得（ローカルParquet優先、なければ気象庁APIへ自動フォールバック）
    if amedas_dir.exists():
        df_amedas = cached_load_amedas(str(amedas_dir), target_iso, extent_tuple)
    else:
        df_amedas = cached_load_amedas_jma_api(target_iso, extent_tuple)

    if show_amedas_temp_shade and not df_amedas.empty:
        plot_amedas_temp_shade(
            ax,
            proj,
            df_amedas,
            extent,
            is_regional,
            sigma=temp_sigma,
            vmin=temp_vmin,
            vmax=temp_vmax,
            cmap=temp_cmap,
        )

    if show_isobars:
        ds_msm = cached_get_msm(dt_str, extent_tuple)
        plot_msm_isobars(ax, proj, ds_msm, is_regional, sigma=msm_sigma)

    sub_obs = (
        cached_get_obs(target_iso, extent_tuple, station_min_dist)
        if show_synop_metar
        else pd.DataFrame()
    )

    df_amedas_plot = df_amedas.copy()
    if not df_amedas_plot.empty and not sub_obs.empty:
        excl_deg = 0.12 if is_regional else 0.30
        keep_mask = [
            np.hypot(sub_obs["lat"] - row["lat"], sub_obs["lon"] - row["lon"]).min()
            >= excl_deg
            for _, row in df_amedas_plot.iterrows()
        ]
        df_amedas_plot = df_amedas_plot[keep_mask]

    if not df_amedas_plot.empty:
        plot_amedas_barbs_and_obs(
            ax,
            proj,
            df_amedas_plot,
            is_regional,
            show_wind=show_amedas_wind,
            show_obs=show_amedas_obs,
        )

    if not sub_obs.empty:
        plot_synop_metar_station(
            ax, proj, sub_obs, show_station_name=show_station_name
        )

    reg_en = REGION_EN.get(region, region)
    utc_str = target_dt.strftime("%Y-%m-%d %H:%M UTC")
    jst_str = target_dt.tz_convert("Asia/Tokyo").strftime("%Y-%m-%d %H:%M JST")
    ax.set_title(
        f"Surface Analysis ({reg_en} / MSM MSLP + AMeDAS + SYNOP/METAR)\nValid: {utc_str} / {jst_str}",
        fontsize=11,
        pad=10,
        fontweight="semibold",
    )
    plt.tight_layout()
    return fig


# --- サイドバー UI ---
st.sidebar.header("解析条件設定")

# 確実にデータ取得できるデフォルト時刻（現在UTCから6時間前を3時間刻みに丸める）
safe_utc = datetime.now(timezone.utc) - timedelta(hours=6)
default_hour = (safe_utc.hour // 3) * 3
hour_options = [0, 3, 6, 9, 12, 15, 18, 21]

sel_date = st.sidebar.date_input("対象日 (UTC)", value=safe_utc.date())
sel_hour = st.sidebar.radio(
    "対象時刻 (UTC)",
    options=hour_options,
    index=hour_options.index(default_hour),
    horizontal=True,
    format_func=lambda h: f"{h:02d}Z",
)

target_dt = pd.Timestamp(datetime.combine(sel_date, time(sel_hour, 0)), tz="UTC")
st.sidebar.caption(
    f"日本時間: **{target_dt.tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M JST')}**"
)

region_options = list(REGION_EXTENTS.keys())
default_idx = region_options.index("東北") if "東北" in region_options else 0
sel_region = st.sidebar.selectbox("表示領域", options=region_options, index=default_idx)

st.sidebar.subheader("表示レイヤー")
show_isobars = st.sidebar.checkbox("MSM 海面更正気圧 (等圧線)", value=True)
show_synop_metar = st.sidebar.checkbox("SYNOP / METAR 国際式プロット", value=True)
show_station_name = st.sidebar.checkbox(
    "SYNOP / METAR 地点名を表示", value=False, disabled=not show_synop_metar
)

amedas_available = DEFAULT_AMEDAS_DIR.exists()
show_amedas_temp_shade = st.sidebar.checkbox(
    "アメダス 気温カラーシェード", value=amedas_available, disabled=not amedas_available
)
show_amedas_obs = st.sidebar.checkbox(
    "アメダス 気温値・降水マーカー", value=amedas_available, disabled=not amedas_available
)
show_amedas_wind = st.sidebar.checkbox(
    "アメダス 風矢羽", value=amedas_available, disabled=not amedas_available
)
if not amedas_available:
    st.sidebar.info("Cloud環境（ローカルParquetなし）のため、MSMとSYNOP/METARを描画します。")

# 気温カラー設定（対象月に応じてデフォルトレンジを自動調整）
with st.sidebar.expander("気温カラー設定", expanded=True):
    m = sel_date.month
    if m in [12, 1, 2]:
        def_min, def_max = -10, 16
    elif m in [3, 4, 11]:
        def_min, def_max = -4, 22
    elif m in [5, 10]:
        def_min, def_max = 4, 28
    else:
        def_min, def_max = 12, 38

    temp_vmin, temp_vmax = st.slider(
        "気温カラーバー範囲 (℃)",
        min_value=-20,
        max_value=42,
        value=(def_min, def_max),
        step=2,
        disabled=not show_amedas_temp_shade,
    )
    temp_cmap = st.selectbox(
        "カラーマップ",
        options=["coolwarm", "RdYlBu_r", "turbo", "viridis"],
        index=0,
        disabled=not show_amedas_temp_shade,
    )

with st.sidebar.expander("平滑化・詳細パラメータ"):
    msm_sigma = st.slider("MSM 等圧線スムージング (sigma)", 0.0, 3.0, 1.0, 0.2)
    temp_sigma = st.slider("アメダス気温スムージング (sigma)", 0.0, 3.0, 1.2, 0.2)

# --- メイン描画エリア ---
fig = create_surface_figure(
    target_dt=target_dt,
    region=sel_region,
    amedas_dir=DEFAULT_AMEDAS_DIR,
    show_isobars=show_isobars,
    show_amedas_temp_shade=show_amedas_temp_shade,
    show_amedas_obs=show_amedas_obs,
    show_amedas_wind=show_amedas_wind,
    show_synop_metar=show_synop_metar,
    show_station_name=show_station_name,
    msm_sigma=msm_sigma,
    temp_sigma=temp_sigma,
    temp_vmin=float(temp_vmin),
    temp_vmax=float(temp_vmax),
    temp_cmap=temp_cmap,
)

st.pyplot(fig, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)

fname = f"surface_analysis_{sel_region}_{target_dt.strftime('%Y%m%d%H')}.png"
st.download_button(
    label="PNG画像をダウンロード",
    data=buf.getvalue(),
    file_name=fname,
    mime="image/png",
)

# --- 気象庁防災情報 JSON からアメダスデータを取得（直近48時間用） ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_jma_amedas_table() -> pd.DataFrame:
    """気象庁の観測所マスタ (amedastable.json) を取得"""
    url = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    raw = res.json()
    records = []
    for stn_id, info in raw.items():
        lat = info["lat"][0] + info["lat"][1] / 60.0
        lon = info["lon"][0] + info["lon"][1] / 60.0
        records.append({"stn_id": stn_id, "lat": lat, "lon": lon})
    return pd.DataFrame(records)


@st.cache_data(ttl=1800, show_spinner="気象庁APIからアメダス実況を取得中...")
def cached_load_amedas_jma_api(target_iso: str, extent_tuple: tuple) -> pd.DataFrame:
    """直近48時間以内であれば気象庁防災情報JSONからアメダス実況を取得"""
    dt_utc = pd.to_datetime(target_iso, utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")

    # 保持期限（約48時間）を超えている場合は空を返す
    if (now_utc - dt_utc).total_seconds() > 47 * 3600 or dt_utc > now_utc:
        return pd.DataFrame()

    dt_jst = dt_utc.tz_convert("Asia/Tokyo")
    ts_str = dt_jst.strftime("%Y%m%d%H0000")
    url = f"https://www.jma.go.jp/bosai/amedas/data/map/{ts_str}.json"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return pd.DataFrame()
        obs_map = res.json()
        df_table = fetch_jma_amedas_table()
    except Exception:
        return pd.DataFrame()

    rows = []
    # 気象庁16方位（1:NNE 〜 16:N, 0:静穏）を数学角・風向(度)へ変換
    for _, stn in df_table.iterrows():
        sid = stn["stn_id"]
        if sid not in obs_map:
            continue
        d = obs_map[sid]

        # [値, 品質フラグ] の形式（フラグ 0 が正常値）
        temp = d["temp"][0] if ("temp" in d and d["temp"][1] == 0) else np.nan
        prec = d["precipitation1h"][0] if ("precipitation1h" in d and d["precipitation1h"][1] == 0) else 0.0
        wspd = d["wind"][0] if ("wind" in d and d["wind"][1] == 0) else np.nan
        wdir16 = d["windDirection"][0] if ("windDirection" in d and d["windDirection"][1] == 0) else 0

        if np.isfinite(wspd) and wdir16 > 0:
            deg = wdir16 * 22.5
            rad = np.deg2rad(deg)
            u = -wspd * np.sin(rad)
            v = -wspd * np.cos(rad)
        elif np.isfinite(wspd) and wdir16 == 0:
            u, v = 0.0, 0.0
        else:
            u, v = np.nan, np.nan

        rows.append({
            "lat": stn["lat"],
            "lon": stn["lon"],
            "temp_val_safe": float(temp) if temp is not None else np.nan,
            "prec_val_safe": float(prec) if prec is not None else 0.0,
            "wspd_safe": float(wspd) if wspd is not None else np.nan,
            "wind_u_safe": float(u),
            "wind_v_safe": float(v),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    lon_min, lon_max, lat_min, lat_max = extent_tuple
    mask = (
        (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
        & (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
    )
    return df[mask].reset_index(drop=True)
