# app/main.py
# Dashboard de lectura de reportes (sin reentrenar, sin inferencias nuevas)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class Artifacts:
    metrics: Path
    pred: Path
    anom: Path
    reco: Path
    f24: Path
    f7d: Path


def repo_root() -> Path:
    # app/main.py -> app/ -> repo_root
    return Path(__file__).resolve().parents[1]


ROOT_DIR = repo_root()
REPORTS_DIR = ROOT_DIR / "reports"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    # lectura robusta (evita que un encoding raro tumbe todo)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def safe_path(path: Path) -> Optional[Path]:
    return path if exists(path) else None


def normalize_sede_id(series: pd.Series) -> pd.Series:
    # evita problemas cuando sede_id llega como int, float o string
    return series.astype(str).str.strip()


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def format_float(x: Any, ndigits: int = 4) -> str:
    if isinstance(x, (int, float)) and pd.notna(x):
        return f"{x:.{ndigits}f}"
    return str(x)


def explain_box() -> None:
    st.markdown(
        """
**Cómo leer este dashboard**
- **yhat**: energía predicha por el modelo (kWh).
- **y** / **energia_total_kwh**: energía real observada (kWh).
- **residual = y - yhat**: error del modelo.
- **abs_residual = |residual|**: magnitud del error.
- **Anomalía**: registros con **abs_residual** por encima del **umbral p99** (percentil 99).
        """.strip()
    )


st.set_page_config(page_title="IAMinds UPTC - Energy", layout="wide")
st.title("IAMinds UPTC – Energía (dashboard de reportes)")
st.caption("Lee artefactos en /reports. No reentrena. No genera inferencias nuevas.")
explain_box()

# --- Rutas de artefactos ---
paths = Artifacts(
    metrics=REPORTS_DIR / "metrics_energy.json",
    pred=REPORTS_DIR / "pred_vs_real_energy.csv",
    anom=REPORTS_DIR / "anomalies_energy.csv",
    reco=REPORTS_DIR / "recommendations_energy.csv",
    f24=REPORTS_DIR / "forecast_24h_energy.csv",
    f7d=REPORTS_DIR / "forecast_7d_energy.csv",
)

p_metrics = safe_path(paths.metrics)
p_pred = safe_path(paths.pred)
p_anom = safe_path(paths.anom)
p_reco = safe_path(paths.reco)
p_f24 = safe_path(paths.f24)
p_f7d = safe_path(paths.f7d)

# --- Cargar lo que exista ---
metrics: dict[str, Any] = {}
if p_metrics:
    metrics = load_json(p_metrics)

f24 = load_csv(p_f24) if p_f24 else None
f7d = load_csv(p_f7d) if p_f7d else None
pv = load_csv(p_pred) if p_pred else None
anom = load_csv(p_anom) if p_anom else None
reco = load_csv(p_reco) if p_reco else None

# --- Barra superior: KPIs rápidos ---
col1, col2, col3, col4, col5 = st.columns(5)

mae = metrics.get("mae")
rmse = metrics.get("rmse")
target = metrics.get("target", "energia_total_kwh")
split_q = metrics.get("temporal_split_quantile")
cutoff = metrics.get("cutoff_timestamp")

with col1:
    st.metric("Target", str(target))
with col2:
    st.metric("MAE", format_float(mae))
with col3:
    st.metric("RMSE", format_float(rmse))
with col4:
    st.metric("Split temporal (q)", format_float(split_q))
with col5:
    st.metric("Cutoff", str(cutoff))

# Last updated (si está disponible o como fallback por mtime del archivo)
last_updated = metrics.get("generated_at")
if not last_updated and p_metrics:
    last_updated = pd.to_datetime(p_metrics.stat().st_mtime, unit="s").isoformat()

st.caption(f"Reports dir: {REPORTS_DIR.as_posix()} | Last updated: {last_updated}")

st.divider()

# --- Sidebar: selector de sede ---
def build_sede_catalog(*dfs: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            continue
        if "sede_id" not in df.columns:
            continue
        tmp = df.copy()
        tmp["sede_id"] = normalize_sede_id(tmp["sede_id"])
        if "sede" in tmp.columns:
            tmp["sede"] = tmp["sede"].astype(str).str.strip()
        else:
            tmp["sede"] = ""
        rows.append(tmp[["sede_id", "sede"]])
    if not rows:
        return pd.DataFrame(columns=["sede_id", "sede"])
    cat = pd.concat(rows, ignore_index=True).dropna(subset=["sede_id"]).drop_duplicates()
    cat = cat.sort_values(["sede", "sede_id"], kind="stable")
    return cat


catalog = build_sede_catalog(f24, f7d, pv, anom, reco)

st.sidebar.header("Filtros")
if len(catalog) == 0:
    sede_id_sel = "(sin sede_id disponible)"
    st.sidebar.warning("No encontré 'sede_id' en los reportes cargados.")
else:
    # opciones mostradas como "id — sede"
    display = []
    for _, r in catalog.iterrows():
        label = f"{r['sede_id']} — {r['sede']}" if r["sede"] else f"{r['sede_id']}"
        display.append(label)

    selected = st.sidebar.selectbox("Selecciona sede", options=["(todas)"] + display)
    sede_id_sel = "(todas)" if selected == "(todas)" else selected.split("—")[0].strip()

show_only_anomalies = st.sidebar.checkbox("Solo anomalías (is_anomaly=True)", value=True)

st.divider()

# --- Tabs para demo ---
tab_overview, tab_forecast, tab_anom_reco, tab_evidence = st.tabs(
    ["Resumen", "Pronósticos", "Anomalías y recomendaciones", "Evidencia real vs pred"]
)

# =========================
# Resumen
# =========================
with tab_overview:
    st.subheader("Estado de artefactos")
    status_rows = [
        ("metrics_energy.json", bool(p_metrics)),
        ("pred_vs_real_energy.csv", bool(p_pred)),
        ("anomalies_energy.csv", bool(p_anom)),
        ("recommendations_energy.csv", bool(p_reco)),
        ("forecast_24h_energy.csv", bool(p_f24)),
        ("forecast_7d_energy.csv", bool(p_f7d)),
    ]
    st.dataframe(
        pd.DataFrame(status_rows, columns=["artefacto", "existe"]),
        width="stretch",
        hide_index=True,
    )

    # Conteos rápidos
    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Filas forecast 24h", f"{len(f24):,}" if isinstance(f24, pd.DataFrame) else "NA")
    with cB:
        st.metric("Filas forecast 7d", f"{len(f7d):,}" if isinstance(f7d, pd.DataFrame) else "NA")
    with cC:
        st.metric("Filas evidencia test", f"{len(pv):,}" if isinstance(pv, pd.DataFrame) else "NA")

    st.markdown("### Interpretabilidad mínima para jurado")
    st.markdown(
        """
- El modelo **predice kWh** por sede.
- La **anomalía** se define por error alto del modelo (**abs_residual**) con umbral **p99**.
- Las **recomendaciones** son reglas operativas asociadas al contexto (horario, fin de semana, severidad).
        """.strip()
    )

# =========================
# Pronósticos
# =========================
with tab_forecast:
    st.subheader("Pronósticos por sede")

    def filter_by_sede(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "sede_id" in out.columns:
            out["sede_id"] = normalize_sede_id(out["sede_id"])
        if sede_id_sel not in ("(todas)", "(sin sede_id disponible)") and "sede_id" in out.columns:
            out = out[out["sede_id"] == sede_id_sel]
        return out

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 24 horas")
        if isinstance(f24, pd.DataFrame):
            df = filter_by_sede(f24)
            cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df.columns]
            st.dataframe(df[cols].head(48), width="stretch")

            if "timestamp" in df.columns and "yhat" in df.columns:
                tmp = df.copy()
                tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
                tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp")
                tmp = tmp.set_index("timestamp")
                st.line_chart(tmp["yhat"].head(200))
        else:
            st.warning("No encuentro forecast_24h_energy.csv en /reports.")

    with c2:
        st.markdown("### 7 días")
        if isinstance(f7d, pd.DataFrame):
            df = filter_by_sede(f7d)
            cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df.columns]
            st.dataframe(df[cols].head(48), width="stretch")

            if "timestamp" in df.columns and "yhat" in df.columns:
                tmp = df.copy()
                tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
                tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp")
                tmp = tmp.set_index("timestamp")
                st.line_chart(tmp["yhat"].head(400))
        else:
            st.warning("No encuentro forecast_7d_energy.csv en /reports.")

# =========================
# Anomalías + Recomendaciones
# =========================
with tab_anom_reco:
    st.subheader("Anomalías y recomendaciones")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Anomalías (top)")
        if isinstance(anom, pd.DataFrame):
            df = anom.copy()
            if "sede_id" in df.columns:
                df["sede_id"] = normalize_sede_id(df["sede_id"])
            if sede_id_sel not in ("(todas)", "(sin sede_id disponible)") and "sede_id" in df.columns:
                df = df[df["sede_id"] == sede_id_sel]

            # Filtrar solo anomalías si la columna existe
            if show_only_anomalies and "is_anomaly" in df.columns:
                # is_anomaly puede venir como bool, 0/1 o string
                df["is_anomaly_norm"] = df["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])
                df = df[df["is_anomaly_norm"]]

            # normalizar severidad
            if "abs_residual" not in df.columns and "residual" in df.columns:
                df["abs_residual"] = df["residual"].abs()

            sev_col = "abs_residual" if "abs_residual" in df.columns else None
            if sev_col:
                df = df.sort_values(sev_col, ascending=False)

            cols = [c for c in ["timestamp", "sede_id", "sede", "energia_total_kwh", "yhat", "residual", "abs_residual", "threshold_p99"] if c in df.columns]
            st.dataframe(df[cols].head(20), width="stretch")

            if len(df) == 0:
                st.info("No hay filas tras aplicar filtros (revisa sede o 'Solo anomalías').")
        else:
            st.warning("No encuentro anomalies_energy.csv en /reports.")

    with c4:
        st.markdown("### Recomendaciones (top)")
        if isinstance(reco, pd.DataFrame):
            df = reco.copy()
            if "sede_id" in df.columns:
                df["sede_id"] = normalize_sede_id(df["sede_id"])
            if sede_id_sel not in ("(todas)", "(sin sede_id disponible)") and "sede_id" in df.columns:
                df = df[df["sede_id"] == sede_id_sel]

            # Si hay severidad, ordenar; si no, mostrar igual
            if "severity" in df.columns:
                df = df.sort_values("severity", ascending=False)

            st.dataframe(df.head(20), width="stretch")

            if len(df) == 0:
                st.info("No hay recomendaciones para este filtro de sede.")
        else:
            st.warning("No encuentro recommendations_energy.csv en /reports.")

# =========================
# Evidencia real vs pred
# =========================
with tab_evidence:
    st.subheader("Evidencia: real vs pred (periodo de test)")
    if isinstance(pv, pd.DataFrame):
        df = pv.copy()
        if "sede_id" in df.columns:
            df["sede_id"] = normalize_sede_id(df["sede_id"])
        if sede_id_sel not in ("(todas)", "(sin sede_id disponible)") and "sede_id" in df.columns:
            df = df[df["sede_id"] == sede_id_sel]

        cols = [c for c in ["timestamp", "sede_id", "sede", "y", "yhat"] if c in df.columns]
        st.dataframe(df[cols].head(80), width="stretch")

        if "timestamp" in df.columns and "y" in df.columns and "yhat" in df.columns:
            tmp = df.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp[["y", "yhat"]].head(400))
    else:
        st.warning("No encuentro pred_vs_real_energy.csv en /reports.")
