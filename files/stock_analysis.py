"""
TSMC vs Quanta Stock Correlation Analysis – Streamlit App
"""

import io
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
STOCKS = {
    "TSMC":   "2330.TW",
    "Quanta": "2382.TW",
}
END_DATE    = datetime.today().strftime("%Y-%m-%d")
START_DATE  = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
ROLL_WINDOW = 30
DB_URL = (
    "mssql+pymssql://sa:7TH5AIxg3N9jBcXsdJqZ4o6V82t10mpv"
    "@43.153.159.36:30147/gemio"
)


# ── Functions ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="抓取股價資料中…")
def fetch_prices() -> pd.DataFrame:
    frames = {}
    for name, ticker in STOCKS.items():
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"Cannot fetch {name} ({ticker})")
        frames[name] = raw["Close"].squeeze()
    df = pd.DataFrame(frames).dropna()
    df.index.name = "Date"
    return df


def save_to_db(df: pd.DataFrame) -> None:
    engine = create_engine(DB_URL)
    df_reset = df.reset_index()
    df_reset["Date"] = df_reset["Date"].astype(str)
    df_reset.to_sql("stock_daily_close", engine, if_exists="replace", index=False)

    corr_val = df["TSMC"].corr(df["Quanta"])
    meta = pd.DataFrame([{
        "start_date":   START_DATE,
        "end_date":     END_DATE,
        "tsmc_rows":    len(df),
        "pearson_corr": round(corr_val, 6),
        "created_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    meta.to_sql("stock_analysis_meta", engine, if_exists="replace", index=False)


def compute_metrics(df: pd.DataFrame):
    pearson = df["TSMC"].corr(df["Quanta"])
    rolling = df["TSMC"].rolling(ROLL_WINDOW).corr(df["Quanta"])
    returns = df.pct_change().dropna()
    return pearson, rolling, returns


def plot_charts(df: pd.DataFrame, pearson: float,
                rolling: pd.Series, returns: pd.DataFrame):
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"TSMC (2330) vs Quanta (2382) Stock Correlation\n"
        f"{START_DATE} ~ {END_DATE}   Pearson r = {pearson:.4f}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    color_tsmc   = "#e8404a"
    color_quanta = "#3a7eca"

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(df.index, df["TSMC"],   color=color_tsmc,   lw=1.5, label="TSMC")
    ax1.set_ylabel("TSMC Close (TWD)", color=color_tsmc, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_tsmc)
    ax1b = ax1.twinx()
    ax1b.plot(df.index, df["Quanta"], color=color_quanta, lw=1.5, label="Quanta")
    ax1b.set_ylabel("Quanta Close (TWD)", color=color_quanta, fontsize=11)
    ax1b.tick_params(axis="y", labelcolor=color_quanta)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.set_title("Daily Close Price", fontsize=12, pad=8)

    ax2 = fig.add_subplot(3, 2, 3)
    ax2.scatter(returns["TSMC"], returns["Quanta"], alpha=0.35, s=12, color="#9b59b6")
    m, b = np.polyfit(returns["TSMC"], returns["Quanta"], 1)
    x_line = np.linspace(returns["TSMC"].min(), returns["TSMC"].max(), 100)
    ax2.plot(x_line, m * x_line + b, color="tomato", lw=1.5, linestyle="--",
             label=f"Trend  y={m:.2f}x+{b:.4f}")
    ax2.axhline(0, color="gray", lw=0.7, linestyle=":")
    ax2.axvline(0, color="gray", lw=0.7, linestyle=":")
    ax2.set_xlabel("TSMC Daily Return", fontsize=10)
    ax2.set_ylabel("Quanta Daily Return", fontsize=10)
    ax2.set_title(f"Daily Return Scatter (r = {pearson:.4f})", fontsize=12, pad=8)
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(3, 2, 4)
    valid_roll = rolling.dropna()
    sns.histplot(valid_roll, bins=40, color="#3a7eca", edgecolor="white", ax=ax3, kde=True)
    ax3.axvline(pearson, color="tomato", lw=2, linestyle="--",
                label=f"Pearson = {pearson:.4f}")
    ax3.set_xlabel(f"{ROLL_WINDOW}-day Rolling Corr", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.set_title(f"{ROLL_WINDOW}-day Rolling Correlation Distribution", fontsize=12, pad=8)
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(3, 1, 3)
    ax4.plot(rolling.index, rolling, color="#2ecc71", lw=1.4,
             label=f"{ROLL_WINDOW}-day Rolling Corr")
    ax4.axhline(pearson, color="tomato", lw=1.5, linestyle="--",
                label=f"Pearson = {pearson:.4f}")
    ax4.axhline(0, color="gray", lw=0.7, linestyle=":")
    ax4.fill_between(rolling.index, rolling, pearson,
                     where=(rolling >= pearson), alpha=0.15, color="#2ecc71")
    ax4.fill_between(rolling.index, rolling, pearson,
                     where=(rolling < pearson),  alpha=0.15, color="tomato")
    ax4.set_ylim(-1, 1)
    ax4.set_ylabel("Correlation", fontsize=11)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax4.set_title(f"{ROLL_WINDOW}-day Rolling Correlation Time Series", fontsize=12, pad=8)
    ax4.legend(loc="lower right", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="股價關聯性分析", layout="wide")
st.title("台積電 vs 廣達 股價關聯性分析")
st.caption(f"資料期間：{START_DATE} ～ {END_DATE}")

if st.button("重新分析", type="primary"):
    st.cache_data.clear()

try:
    df = fetch_prices()
    pearson, rolling, returns = compute_metrics(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pearson 相關係數", f"{pearson:.4f}")
    col2.metric("資料筆數", f"{len(df)} 個交易日")
    col3.metric("30日滾動相關（最新）", f"{rolling.dropna().iloc[-1]:.4f}")

    with st.spinner("存入資料庫…"):
        save_to_db(df)
    st.success("資料已存入 SQL Server（gemio.stock_daily_close）")

    st.pyplot(plot_charts(df, pearson, rolling, returns))

except Exception as exc:
    st.error(f"錯誤：{exc}")
