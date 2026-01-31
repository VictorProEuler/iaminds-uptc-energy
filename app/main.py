from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def try_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def try_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


st.set_page_config(page_title="IA Minds - UPTC Energy", layout="wide")
st.title("IA Minds – UPTC: Predicción, anomalías y recomendaciones")
st.caption(f"Leyendo outputs desde: {REPORTS_DIR}")

metrics = try_read_json(REPORTS_DIR / "metrics_energy.json")
pred = try_read_csv(REPORTS_DIR / "pred_vs_real_energy.csv")
anoms = try_read_csv(REPORTS_DIR / "anomalies_energy.csv")
recs = try_read_csv(REPORTS_DIR / "recommendations_energy.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Métricas")
    if metrics is None:
        st.info("Aún no hay metrics_energy.json en reports/.")
    else:
        st.json(metrics)

with col2:
    st.subheader("Anomalías (top 20)")
    if anoms is None:
        st.info("Aún no hay anomalies_energy.csv en reports/.")
    else:
        st.dataframe(anoms.head(20), use_container_width=True)

with col3:
    st.subheader("Recomendaciones (top 20)")
    if recs is None:
        st.info("Aún no hay recommendations_energy.csv en reports/.")
    else:
        st.dataframe(recs.head(20), use_container_width=True)

st.divider()
st.subheader("Real vs Predicho")
if pred is None:
    st.info("Aún no hay pred_vs_real_energy.csv en reports/.")
else:
    st.dataframe(pred.head(50), use_container_width=True)

