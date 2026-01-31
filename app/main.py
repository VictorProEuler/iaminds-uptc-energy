# app/main.py
# Dashboard de lectura de reportes (sin reentrenar, sin inferencias nuevas)
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


REPORTS_DIR = Path("reports")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_read(path: Path) -> Optional[Path]:
    return path if path.exists() else None


st.set_page_config(page_title="IAMinds UPTC - Energy", layout="wide")

st.title("IAMinds UPTC – Energía (dashboard de reportes)")
st.caption("Este tablero lee los artefactos exportados en /reports. No recalcula el modelo.")

# --- Cargar artefactos si existen ---
p_metrics = safe_read(REPORTS_DIR / "metrics_energy.json")
p_pred = safe_read(REPORTS_DIR / "pred_vs_real_energy.csv")
p_anom = safe_read(REPORTS_DIR / "anomalies_energy.csv")
p_reco = safe_read(REPORTS_DIR / "recommendations_energy.csv")
p_f24 = safe_read(REPORTS_DIR / "forecast_24h_energy.csv")
p_f7d = safe_read(REPORTS_DIR / "forecast_7d_energy.csv")

colA, colB, colC = st.columns(3)

with colA:
    st.subheader("Métricas")
    if p_metrics:
        metrics = load_json(p_metrics)
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        st.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae))
        st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse))
        st.caption(f"Archivo: {p_metrics.as_posix()}")
    else:
        st.warning("No encuentro metrics_energy.json en /reports")

with colB:
    st.subheader("Pronóstico 24h")
    if p_f24:
        f24 = load_csv(p_f24)
        st.caption(f"Filas: {len(f24):,} | Archivo: {p_f24.as_posix()}")
    else:
        f24 = None
        st.warning("No encuentro forecast_24h_energy.csv")

with colC:
    st.subheader("Pronóstico 7d")
    if p_f7d:
        f7d = load_csv(p_f7d)
        st.caption(f"Filas: {len(f7d):,} | Archivo: {p_f7d.as_posix()}")
    else:
        f7d = None
        st.warning("No encuentro forecast_7d_energy.csv")

st.divider()

# --- Sidebar: sede ---
sede_options = []
for df_ in [f24, f7d]:
    if isinstance(df_, pd.DataFrame):
        if "sede_id" in df_.columns:
            sede_options = sorted(df_["sede_id"].dropna().unique().tolist())
            break

if not sede_options and p_pred:
    tmp = load_csv(p_pred)
    if "sede_id" in tmp.columns:
        sede_options = sorted(tmp["sede_id"].dropna().unique().tolist())

sede_sel = st.sidebar.selectbox("Selecciona sede_id", options=sede_options if sede_options else ["(sin sede_id disponible)"])

# --- Sección: Forecasts ---
st.subheader("Pronósticos por sede")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 24 horas")
    if f24 is not None and sede_sel != "(sin sede_id disponible)":
        df = f24.copy()
        if "sede_id" in df.columns:
            df = df[df["sede_id"] == sede_sel]
        # columnas esperadas: timestamp, yhat
        cols = [c for c in ["timestamp", "yhat", "sede", "sede_id"] if c in df.columns]
        st.dataframe(df[cols].head(24), use_container_width=True)
        if "yhat" in df.columns:
            st.line_chart(df["yhat"].head(24))
    else:
        st.info("Carga forecast_24h_energy.csv para ver esta sección.")

with c2:
    st.markdown("### 7 días")
    if f7d is not None and sede_sel != "(sin sede_id disponible)":
        df = f7d.copy()
        if "sede_id" in df.columns:
            df = df[df["sede_id"] == sede_sel]
        cols = [c for c in ["timestamp", "yhat", "sede", "sede_id"] if c in df.columns]
        st.dataframe(df[cols].head(24), use_container_width=True)
        if "yhat" in df.columns:
            st.line_chart(df["yhat"].head(24))
    else:
        st.info("Carga forecast_7d_energy.csv para ver esta sección.")

st.divider()

# --- Sección: Anomalías + Recomendaciones ---
st.subheader("Anomalías y recomendaciones (por sede)")

c3, c4 = st.columns(2)

with c3:
    st.markdown("### Anomalías (top)")
    if p_anom:
        anom = load_csv(p_anom)
        df = anom.copy()
        if sede_sel != "(sin sede_id disponible)" and "sede_id" in df.columns:
            df = df[df["sede_id"] == sede_sel]

        # normalizar severidad
        if "abs_residual" in df.columns:
            sev_col = "abs_residual"
        elif "residual" in df.columns:
            sev_col = "residual"
            df["abs_residual"] = df["residual"].abs()
            sev_col = "abs_residual"
        else:
            sev_col = None

        if sev_col:
            df = df.sort_values(sev_col, ascending=False)

        cols = [c for c in ["timestamp", "sede_id", "sede", "energia_total_kwh", "yhat", "residual", "abs_residual"] if c in df.columns]
        st.dataframe(df[cols].head(15), use_container_width=True)
    else:
        st.warning("No encuentro anomalies_energy.csv")

with c4:
    st.markdown("### Recomendaciones (top)")
    if p_reco:
        reco = load_csv(p_reco)
        df = reco.copy()
        if sede_sel != "(sin sede_id disponible)" and "sede_id" in df.columns:
            df = df[df["sede_id"] == sede_sel]
        st.dataframe(df.head(15), use_container_width=True)
    else:
        st.warning("No encuentro recommendations_energy.csv")

st.divider()

# --- Sección: Evidencia real vs pred ---
st.subheader("Evidencia: real vs pred (periodo de test)")
if p_pred:
    pv = load_csv(p_pred)
    df = pv.copy()
    if sede_sel != "(sin sede_id disponible)" and "sede_id" in df.columns:
        df = df[df["sede_id"] == sede_sel]

    cols = [c for c in ["timestamp", "sede_id", "sede", "y", "yhat"] if c in df.columns]
    st.dataframe(df[cols].head(50), use_container_width=True)

    # chart simple
    if "y" in df.columns and "yhat" in df.columns:
        st.line_chart(df[["y", "yhat"]].head(200))
else:
    st.info("No encuentro pred_vs_real_energy.csv")
