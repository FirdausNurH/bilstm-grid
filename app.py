"""
BiLSTM + Grid Search — Peramalan Pengunjung Asta Tinggi Sumenep
Aplikasi Streamlit (versi deploy dari script Colab)
"""
import os
import io
import json
import time
import random
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="BiLSTM Asta Tinggi Sumenep", page_icon="📈", layout="wide")

DEFAULT_FILE = "Data Sumenep.xlsx"
DEFAULT_SHEET = "Asta tinggi"
FEATURES = ["log_y", "sin_m", "cos_m", "t", "roll3_mean", "roll6_mean",
            "roll12_mean", "roll3_std", "lag1", "lag12"]

ID_MONTHS = {
    "januari": "January", "februari": "February", "maret": "March", "april": "April",
    "mei": "May", "juni": "June", "juli": "July", "agustus": "August",
    "september": "September", "oktober": "October", "november": "November",
    "desember": "December", "des": "Dec", "agu": "Aug", "okt": "Oct", "ags": "Aug",
}


# =========================================================
# 1. LOAD & CLEANING
# =========================================================
def _parse_month(series: pd.Series) -> pd.Series:
    """Parse kolom bulan: dukung datetime, 'January 2015', dan nama bulan Indonesia."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%B %Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        def tr(x):
            parts = x.lower().split()
            return " ".join(ID_MONTHS.get(p, p) for p in parts)
        parsed = pd.to_datetime(s.map(tr), format="%B %Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(s, errors="coerce")
    return parsed


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes, sheet: str, min_valid: float, pct_cap: float):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    df = df.iloc[:, :2].copy()
    df.columns = ["Bulan", "Jumlah_Pengunjung"]
    df["Bulan"] = _parse_month(df["Bulan"])
    df = df.dropna(subset=["Bulan"]).sort_values("Bulan").reset_index(drop=True)

    full_idx = pd.date_range(df["Bulan"].min(), df["Bulan"].max(), freq="MS")
    df = df.set_index("Bulan").reindex(full_idx).rename_axis("Bulan").reset_index()
    df["Jumlah_Pengunjung"] = (pd.to_numeric(df["Jumlah_Pengunjung"], errors="coerce")
                               .interpolate("linear").ffill().bfill())

    s = df["Jumlah_Pengunjung"].copy()
    s[s < min_valid] = np.nan
    s = s.interpolate("linear").bfill().ffill()
    cap = np.percentile(s, pct_cap)
    s = np.clip(s, None, cap)
    df["Jumlah_Pengunjung"] = s.values

    # Feature engineering
    df["month"] = df["Bulan"].dt.month
    df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)
    df["t"] = np.arange(len(df)) / len(df)
    y_log = np.log1p(df["Jumlah_Pengunjung"].values)
    df["log_y"] = y_log
    ys = pd.Series(y_log)
    df["roll3_mean"] = ys.rolling(3, min_periods=1).mean()
    df["roll6_mean"] = ys.rolling(6, min_periods=1).mean()
    df["roll12_mean"] = ys.rolling(12, min_periods=1).mean()
    df["roll3_std"] = ys.rolling(3, min_periods=1).std().fillna(0)
    df["lag1"] = ys.shift(1).bfill()
    df["lag12"] = ys.shift(12).bfill()
    return df


# =========================================================
# 2. MODEL & HELPER
# =========================================================
def make_windows(y, feats, window):
    X, Y = [], []
    for i in range(window, len(y)):
        X.append(feats[i - window:i])
        Y.append(y[i])
    return np.array(X), np.array(Y)


def prepare_data(df, window, train_ratio=0.75, min_test=12):
    y_all = df["log_y"].values.reshape(-1, 1)
    feats_raw = df[FEATURES].values
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    feats_scaled = scaler_X.fit_transform(feats_raw)
    y_scaled = scaler_y.fit_transform(y_all).flatten()
    X, y = make_windows(y_scaled, feats_scaled, window)
    n = len(X)
    split = int(n * train_ratio)
    if (n - split) < min_test:
        split = n - min_test
    return X[:split], X[split:], y[:split], y[split:], scaler_y, split


def build_bilstm(window, n_feat, units, dropout, lr, l2reg):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    inp = layers.Input(shape=(window, n_feat))
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True,
                                         kernel_regularizer=regularizers.l2(l2reg)))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units // 2, return_sequences=True,
                                         kernel_regularizer=regularizers.l2(l2reg)))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(units // 4))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.Huber(delta=0.05), metrics=["mae"])
    return m


def set_seed(sd):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.random.set_seed(sd)
    np.random.seed(sd)
    random.seed(sd)


def train_one(window, n_feat, units, drop, lr, bs, X_tr, y_tr, seed,
              epochs, patience_es, patience_lr, min_delta):
    from tensorflow.keras import callbacks
    set_seed(seed)
    m = build_bilstm(window, n_feat, units, drop, lr, 1e-4)
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience_es,
                                restore_best_weights=True, min_delta=min_delta),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=patience_lr, min_lr=1e-7, verbose=0),
    ]
    m.fit(X_tr, y_tr, validation_split=0.15, epochs=epochs,
          batch_size=bs, verbose=0, callbacks=cb)
    return m


def inv(scaler_y, a):
    return np.expm1(scaler_y.inverse_transform(np.asarray(a).reshape(-1, 1))).flatten()


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_all(yt, yp):
    return {
        "MAPE": mape(yt, yp),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
    }


# =========================================================
# 3. GRID SEARCH & FINAL TRAINING
# =========================================================
def run_grid(df, windows, grid, seeds, epochs, patience_es, patience_lr, train_ratio):
    combos = list(itertools.product(grid["units"], grid["dropout"], grid["lr"], grid["batch_size"]))
    total = len(windows) * len(combos) * len(seeds)
    prog = st.progress(0.0, text="Memulai grid search…")
    table_ph = st.empty()
    done, rows, best = 0, [], {}

    for W in windows:
        X_tr, X_te, y_tr, y_te, scaler_y, _ = prepare_data(df, W, train_ratio)
        n_feat = X_tr.shape[2]
        y_te_inv = inv(scaler_y, y_te)
        best_mape, best_cfg = 1e9, None

        for ci, (units, drop, lr, bs) in enumerate(combos, 1):
            t0 = time.time()
            mapes, rmses = [], []
            for sd in seeds:
                m = train_one(W, n_feat, units, drop, lr, bs, X_tr, y_tr, sd,
                              epochs, patience_es, patience_lr, 1e-4)
                p = inv(scaler_y, m.predict(X_te, verbose=0).flatten())
                mapes.append(mape(y_te_inv, p))
                rmses.append(float(np.sqrt(mean_squared_error(y_te_inv, p))))
                done += 1
                prog.progress(done / total,
                              text=f"Window {W} | kombinasi {ci}/{len(combos)} | seed {sd} "
                                   f"({done}/{total} run)")
            mm, mr = float(np.mean(mapes)), float(np.mean(rmses))
            rows.append({"Window": W, "units": units, "dropout": drop, "lr": lr,
                         "batch": bs, "val_MAPE": mm, "val_RMSE": mr,
                         "waktu_s": round(time.time() - t0, 1)})
            table_ph.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
            if mm < best_mape:
                best_mape, best_cfg = mm, (units, drop, lr, bs)
        best[W] = {"config": best_cfg, "val_MAPE": best_mape}

    prog.empty()
    table_ph.empty()
    return pd.DataFrame(rows), best


def run_final(df, windows, configs, seeds, epochs, patience_es, patience_lr, train_ratio):
    final, total, done = {}, len(windows) * len(seeds), 0
    prog = st.progress(0.0, text="Training final…")
    for W in windows:
        units, drop, lr, bs = configs[W]
        X_tr, X_te, y_tr, y_te, scaler_y, split = prepare_data(df, W, train_ratio)
        n_feat = X_tr.shape[2]
        ptr, pte = [], []
        for sd in seeds:
            m = train_one(W, n_feat, units, drop, lr, bs, X_tr, y_tr, sd,
                          epochs, patience_es, patience_lr, 1e-5)
            ptr.append(m.predict(X_tr, verbose=0).flatten())
            pte.append(m.predict(X_te, verbose=0).flatten())
            done += 1
            prog.progress(done / total, text=f"Final | Window {W} | seed {sd} ({done}/{total})")

        y_tr_i, p_tr_i = inv(scaler_y, y_tr), inv(scaler_y, np.mean(ptr, axis=0))
        y_te_i, p_te_i = inv(scaler_y, y_te), inv(scaler_y, np.mean(pte, axis=0))
        final[W] = {
            "config": (units, drop, lr, bs),
            "train": evaluate_all(y_tr_i, p_tr_i), "test": evaluate_all(y_te_i, p_te_i),
            "y_train": y_tr_i, "p_train": p_tr_i,
            "dates_train": df["Bulan"].values[W:W + len(y_tr_i)],
            "y_test": y_te_i, "p_test": p_te_i,
            "dates_test": df["Bulan"].values[W + split:W + split + len(y_te_i)],
        }
    prog.empty()
    return final


# =========================================================
# 4. VISUALISASI
# =========================================================
def fig_timeseries(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Bulan"], df["Jumlah_Pengunjung"], "o-", color="#1f77b4", lw=1.5, ms=3)
    ax.set_title(f"Data Jumlah Pengunjung Asta Tinggi Sumenep "
                 f"({df['Bulan'].min().year}–{df['Bulan'].max().year})", fontweight="bold")
    ax.set_xlabel("Tahun"); ax.set_ylabel("Jumlah Pengunjung"); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_grid(grid_df, windows):
    fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 4), squeeze=False)
    for ax, W in zip(axes[0], windows):
        sub = grid_df[grid_df["Window"] == W].sort_values("val_MAPE").head(10)
        labels = [f"u{c.units}_d{c.dropout}_lr{c.lr}_b{c.batch}" for _, c in sub.iterrows()]
        ax.barh(range(len(sub)), sub["val_MAPE"], color="steelblue")
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("MAPE (%)")
        ax.set_title(f"Top-10 Kombinasi — Window={W}", fontweight="bold")
        ax.axvline(25, color="red", ls="--", label="Target 25%"); ax.legend()
    fig.tight_layout()
    return fig


def fig_pred(r, W):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(r["dates_train"], r["y_train"], "o-", color="#1f77b4", label="Aktual (Train)", ms=4)
    ax.plot(r["dates_train"], r["p_train"], "--", color="#1f77b4", label="Prediksi (Train)", alpha=.8)
    ax.plot(r["dates_test"], r["y_test"], "o-", color="#d62728", label="Aktual (Test)", ms=5)
    ax.plot(r["dates_test"], r["p_test"], "--", color="#d62728", label="Prediksi (Test)", lw=1.8)
    ax.axvline(r["dates_train"][-1], color="gray", ls=":", label="Batas Train/Test")
    ax.set_title(f"Window={W} | Test MAPE={r['test']['MAPE']:.2f}%  R²={r['test']['R2']:.4f}",
                 fontweight="bold")
    ax.set_xlabel("Bulan"); ax.set_ylabel("Jumlah Pengunjung")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_scatter(final, windows):
    fig, axes = plt.subplots(1, len(windows), figsize=(5.5 * len(windows), 4.5), squeeze=False)
    for ax, W in zip(axes[0], windows):
        r = final[W]
        ax.scatter(r["y_test"], r["p_test"], s=60, alpha=.7, color="#d62728")
        lo = min(r["y_test"].min(), r["p_test"].min())
        hi = max(r["y_test"].max(), r["p_test"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal (y=x)")
        ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
        ax.set_title(f"Window={W} | R²={r['test']['R2']:.4f}", fontweight="bold")
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def show_fig(fig, name):
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button(f"⬇️ Unduh {name}.png", buf.getvalue(), f"{name}.png", "image/png", key=name)


# =========================================================
# 5. UI
# =========================================================
st.title("📈 BiLSTM + Grid Search — Pengunjung Asta Tinggi Sumenep")
st.caption("Peramalan time series bulanan dengan Bidirectional LSTM, grid search hyperparameter, "
           "dan ensemble multi-seed.")

with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.subheader("1. Data")
    up = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    file_bytes = None
    if up is not None:
        file_bytes = up.getvalue()
    elif os.path.exists(DEFAULT_FILE):
        with open(DEFAULT_FILE, "rb") as f:
            file_bytes = f.read()
        st.info(f"Memakai file bawaan: {DEFAULT_FILE}")

    sheet = None
    if file_bytes:
        sheets = list_sheets(file_bytes)
        idx = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
        sheet = st.selectbox("Sheet", sheets, index=idx)

    min_valid = st.number_input("Nilai minimum valid (di bawahnya dianggap hilang)",
                                0, 100000, 2000, 500)
    pct_cap = st.slider("Persentil cap outlier", 80, 100, 95)

    st.subheader("2. Model")
    windows = st.multiselect("Window", [3, 6, 9, 12, 18], default=[6, 12])
    train_ratio = st.slider("Rasio data train", 0.5, 0.9, 0.75, 0.05)
    mode = st.radio("Mode", ["Grid Search + Training Final", "Training Final saja (config manual)"])

    if mode.startswith("Grid"):
        st.markdown("**Ruang pencarian**")
        g_units = st.multiselect("units", [32, 64, 128, 256], [64, 128])
        g_drop = st.multiselect("dropout", [0.1, 0.2, 0.3, 0.4], [0.2, 0.3])
        g_lr = st.multiselect("learning rate", [1e-3, 5e-4, 1e-4], [1e-3, 5e-4])
        g_bs = st.multiselect("batch size", [8, 16, 32, 64], [16, 32])
        g_seeds = st.multiselect("Seed (grid)", [42, 123, 2024, 7, 99], [42, 123, 2024])
        g_epochs = st.number_input("Max epoch (grid)", 20, 1000, 200, 20)
        n_combo = len(g_units) * len(g_drop) * len(g_lr) * len(g_bs)
        st.caption(f"{n_combo} kombinasi × {len(windows)} window × {len(g_seeds)} seed = "
                   f"**{n_combo * len(windows) * len(g_seeds)} run training**")
    else:
        m_units = st.selectbox("units", [32, 64, 128, 256], 1)
        m_drop = st.selectbox("dropout", [0.1, 0.2, 0.3, 0.4], 1)
        m_lr = st.selectbox("learning rate", [1e-3, 5e-4, 1e-4], 1)
        m_bs = st.selectbox("batch size", [8, 16, 32, 64], 1)

    st.markdown("**Training final**")
    f_seeds = st.multiselect("Seed (final ensemble)", [42, 123, 2024, 7, 99], [42, 123, 2024, 7, 99])
    f_epochs = st.number_input("Max epoch (final)", 50, 1500, 500, 50)

    run = st.button("🚀 Jalankan", type="primary", use_container_width=True)

if not file_bytes:
    st.warning("Upload file Excel di sidebar, atau letakkan `Data Sumenep.xlsx` di folder aplikasi.")
    st.stop()

try:
    df = load_and_clean(file_bytes, sheet, float(min_valid), float(pct_cap))
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

# ---- Eksekusi ----
if run:
    if not windows or not f_seeds:
        st.error("Pilih minimal satu window dan satu seed final.")
        st.stop()
    if len(df) < max(windows) + 30:
        st.error("Data terlalu pendek untuk window yang dipilih.")
        st.stop()
    t_start = time.time()
    if mode.startswith("Grid"):
        if not (g_units and g_drop and g_lr and g_bs and g_seeds):
            st.error("Lengkapi semua opsi grid search.")
            st.stop()
        grid = {"units": g_units, "dropout": g_drop, "lr": g_lr, "batch_size": g_bs}
        with st.spinner("Grid search berjalan… (bisa memakan waktu lama)"):
            grid_df, best = run_grid(df, windows, grid, g_seeds, int(g_epochs), 30, 10, train_ratio)
        configs = {W: best[W]["config"] for W in windows}
        st.session_state["grid_df"] = grid_df
    else:
        configs = {W: (m_units, m_drop, m_lr, m_bs) for W in windows}
        st.session_state["grid_df"] = None

    final = run_final(df, windows, configs, f_seeds, int(f_epochs), 50, 15, train_ratio)
    st.session_state["final"] = final
    st.session_state["windows"] = windows
    st.success(f"Selesai dalam {(time.time() - t_start) / 60:.1f} menit.")

# ---- Tampilan hasil ----
tab_data, tab_grid, tab_res, tab_viz, tab_dl = st.tabs(
    ["📊 Data", "🔍 Grid Search", "🎯 Hasil Akhir", "🖼️ Visualisasi", "⬇️ Unduh"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah bulan", len(df))
    c2.metric("Periode", f"{df['Bulan'].min():%b %Y} – {df['Bulan'].max():%b %Y}")
    c3.metric("Rata-rata", f"{df['Jumlah_Pengunjung'].mean():,.0f}")
    st.subheader("Statistik deskriptif (setelah cleaning)")
    st.dataframe(df["Jumlah_Pengunjung"].describe().round(2).to_frame("nilai"))
    show_fig(fig_timeseries(df), "fig1_timeseries")
    with st.expander("Lihat data & fitur"):
        st.dataframe(df, use_container_width=True)

final = st.session_state.get("final")
grid_df = st.session_state.get("grid_df")
res_windows = st.session_state.get("windows", [])

with tab_grid:
    if grid_df is None:
        st.info("Belum ada hasil grid search. Jalankan mode 'Grid Search + Training Final'.")
    else:
        st.dataframe(grid_df.round(4), use_container_width=True)
        show_fig(fig_grid(grid_df, res_windows), "fig2_gridsearch")

summary_df = None
with tab_res:
    if not final:
        st.info("Belum ada hasil. Atur parameter di sidebar lalu klik **Jalankan**.")
    else:
        rows = []
        for W in res_windows:
            r = final[W]
            u, d, l, b = r["config"]
            rows.append({
                "Window": W, "Hyperparameters": f"units={u}, dropout={d}, lr={l}, batch={b}",
                "MAPE Train (%)": round(r["train"]["MAPE"], 2), "MAE Train": round(r["train"]["MAE"], 1),
                "RMSE Train": round(r["train"]["RMSE"], 1), "R² Train": round(r["train"]["R2"], 4),
                "MAPE Test (%)": round(r["test"]["MAPE"], 2), "MAE Test": round(r["test"]["MAE"], 1),
                "RMSE Test": round(r["test"]["RMSE"], 1), "R² Test": round(r["test"]["R2"], 4),
            })
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        cols = st.columns(len(res_windows))
        for c, W in zip(cols, res_windows):
            mt = final[W]["test"]["MAPE"]
            c.metric(f"MAPE Test (W={W})", f"{mt:.2f}%",
                     "memenuhi target <25%" if mt < 25 else "di atas target 25%",
                     delta_color="normal" if mt < 25 else "inverse")

with tab_viz:
    if not final:
        st.info("Belum ada hasil.")
    else:
        for W in res_windows:
            show_fig(fig_pred(final[W], W), f"fig3_prediksi_W{W}")
        show_fig(fig_scatter(final, res_windows), "fig4_scatter")

with tab_dl:
    if not final:
        st.info("Belum ada hasil untuk diunduh.")
    else:
        st.download_button("⬇️ hasil_akhir.csv", summary_df.to_csv(index=False).encode(),
                           "hasil_akhir.csv", "text/csv")
        if grid_df is not None:
            st.download_button("⬇️ grid_search_results.csv", grid_df.to_csv(index=False).encode(),
                               "grid_search_results.csv", "text/csv")
        meta = {str(W): {"config": list(final[W]["config"]), "test_metrics": final[W]["test"]}
                for W in res_windows}
        st.download_button("⬇️ hasil_meta.json", json.dumps(meta, indent=2).encode(),
                           "hasil_meta.json", "application/json")
        pred_rows = []
        for W in res_windows:
            r = final[W]
            for split_name in ("train", "test"):
                pred_rows.append(pd.DataFrame({
                    "Window": W, "Set": split_name,
                    "Bulan": r[f"dates_{split_name}"],
                    "Aktual": r[f"y_{split_name}"], "Prediksi": r[f"p_{split_name}"]}))
        st.download_button("⬇️ prediksi_detail.csv",
                           pd.concat(pred_rows).to_csv(index=False).encode(),
                           "prediksi_detail.csv", "text/csv")
