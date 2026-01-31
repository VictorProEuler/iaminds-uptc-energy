# app/main.py
# Dashboard de lectura de reportes (sin reentrenar, sin inferencias nuevas)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


# -----------------------------
# Config y utilidades
# -----------------------------
@dataclass(frozen=True)
class Artifacts:
    metrics: Path
    pred: Path
    anom: Path
    reco: Path
    f24: Path
    f7d: Path


def repo_root() -> Path:
    # app/main.py -> app/ -> repo root
    return Path(__file__).resolve().parents[1]


ROOT_DIR = repo_root()
REPORTS_DIR = ROOT_DIR / "reports"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def safe_path(path: Path) -> Optional[Path]:
    return path if path.exists() and path.is_file() else None


def normalize_sede_id(series: pd.Series) -> pd.Series:
    # Unifica UPTC-CHI y UPTC_CHI y variantes de casing/espacios
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace("-", "_", regex=False)
    return s


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def format_float(x: Any, ndigits: int = 4) -> str:
    if isinstance(x, (int, float)) and pd.notna(x):
        return f"{x:.{ndigits}f}"
    return str(x)


def inject_css() -> None:
    st.markdown(
        """
<style>
/* Layout general */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14,62,120,0.06) 0%, rgba(14,62,120,0.02) 55%, rgba(0,0,0,0) 100%);
  border-right: 1px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stCheckbox {
  padding: 0.2rem 0.2rem 0.4rem 0.2rem;
}

/* Tarjetas */
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  background: rgba(255,255,255,0.65);
  box-shadow: 0 6px 22px rgba(0,0,0,0.04);
}
.card-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.2rem; }
.card-value { font-size: 1.45rem; font-weight: 700; margin: 0; }
.badge {
  display: inline-block;
  padding: 0.12rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(14,62,120,0.06);
}

/* Separadores suaves */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.07); margin: 1.3rem 0; }

/* Dataframes */
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(0,0,0,0.07); }
</style>
        """.strip(),
        unsafe_allow_html=True,
    )


def card(col, title: str, value: str, badge: Optional[str] = None) -> None:
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    col.markdown(
        f"""
<div class="card">
  <div class="card-title">{title} {badge_html}</div>
  <p class="card-value">{value}</p>
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="IAMinds UPTC - Energía", layout="wide")
inject_css()

st.title("IAMinds UPTC – Energía")
st.caption("Dashboard de lectura de reportes en /reports (sin reentrenar).")

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

metrics: dict[str, Any] = load_json(p_metrics) if p_metrics else {}

f24 = load_csv(p_f24) if p_f24 else None
f7d = load_csv(p_f7d) if p_f7d else None
pv = load_csv(p_pred) if p_pred else None
anom = load_csv(p_anom) if p_anom else None
reco = load_csv(p_reco) if p_reco else None

# Normalizar sede_id en todos los DF que lo tengan (evita el problema "-" vs "_")
for df in [f24, f7d, pv, anom, reco]:
    if isinstance(df, pd.DataFrame) and "sede_id" in df.columns:
        df["sede_id"] = normalize_sede_id(df["sede_id"])

# KPIs arriba
target = metrics.get("target", "energia_total_kwh")
mae = metrics.get("mae")
rmse = metrics.get("rmse")
cutoff = metrics.get("cutoff_timestamp")

split_method = metrics.get("split_method", "")
split_q = None
if isinstance(split_method, str) and "quantile_" in split_method:
    try:
        split_q = float(split_method.split("quantile_")[-1])
    except ValueError:
        split_q = None

last_updated = metrics.get("generated_at")
if not last_updated and p_metrics:
    last_updated = pd.to_datetime(p_metrics.stat().st_mtime, unit="s").isoformat()

k1, k2, k3, k4, k5 = st.columns(5)
card(k1, "Target", str(target))
card(k2, "MAE", format_float(mae))
card(k3, "RMSE", format_float(rmse))
card(k4, "Split temporal", format_float(split_q), badge="q")
card(k5, "Cutoff", str(cutoff))

st.caption(f"Reports: {REPORTS_DIR.as_posix()} | Last updated: {last_updated}")
st.markdown("<hr/>", unsafe_allow_html=True)

with st.expander("Glosario (qué significa cada campo)"):
    st.markdown(
        """
- **yhat**: energía predicha por el modelo (kWh).
- **y** / **energia_total_kwh**: energía real observada (kWh).
- **residual = y - yhat**: error del modelo.
- **abs_residual = |residual|**: magnitud del error.
- **Anomalía**: registros con **abs_residual** por encima del umbral **p99**.
        """.strip()
    )

# Sidebar: filtros
st.sidebar.header("Filtros")

def build_catalog(*dfs: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            continue
        if "sede_id" not in df.columns:
            continue
        tmp = df[["sede_id"]].copy()
        if "sede" in df.columns:
            tmp["sede"] = df["sede"].astype(str).str.strip()
        else:
            tmp["sede"] = ""
        rows.append(tmp.dropna(subset=["sede_id"]))
    if not rows:
        return pd.DataFrame(columns=["sede_id", "sede"])
    cat = pd.concat(rows, ignore_index=True).drop_duplicates()
    cat = cat.sort_values(["sede", "sede_id"], kind="stable")
    return cat


catalog = build_catalog(f24, f7d, pv, anom, reco)

if len(catalog) == 0:
    sede_id_sel = "(todas)"
    st.sidebar.warning("No encontré 'sede_id' en los reportes cargados.")
else:
    label_to_id: dict[str, str] = {"(todas)": "(todas)"}
    for _, r in catalog.iterrows():
        sid = str(r["sede_id"])
        name = str(r.get("sede", "")).strip()
        label = f"{sid} — {name}" if name and name != "nan" else sid
        label_to_id[label] = sid

    selected_label = st.sidebar.selectbox("Selecciona sede", options=list(label_to_id.keys()))
    sede_id_sel = label_to_id[selected_label]

show_only_anomalies = st.sidebar.checkbox("Solo anomalías (is_anomaly=True)", value=True)

def filter_sede(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sede_id_sel not in ("(todas)",) and "sede_id" in out.columns:
        out = out[out["sede_id"] == sede_id_sel]
    return out


# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Resumen", "Pronósticos", "Anomalías y recomendaciones", "Evidencia real vs pred"]
)

# -----------------------------
# Resumen
# -----------------------------
with tab1:
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

    cA, cB, cC = st.columns(3)
    card(cA, "Filas forecast 24h", f"{len(f24):,}" if isinstance(f24, pd.DataFrame) else "NA")
    card(cB, "Filas forecast 7d", f"{len(f7d):,}" if isinstance(f7d, pd.DataFrame) else "NA")
    card(cC, "Filas evidencia test", f"{len(pv):,}" if isinstance(pv, pd.DataFrame) else "NA")

# -----------------------------
# Pronósticos
# -----------------------------
with tab2:
    st.subheader("Pronósticos")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 24 horas")
        if isinstance(f24, pd.DataFrame):
            df = filter_sede(f24)
            cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df.columns]
            st.dataframe(df[cols].head(48), width="stretch")

            if "timestamp" in df.columns and "yhat" in df.columns:
                tmp = df.copy()
                tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
                tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
                st.line_chart(tmp["yhat"].head(200))
        else:
            st.warning("No encuentro forecast_24h_energy.csv en /reports.")

    with c2:
        st.markdown("### 7 días")
        if isinstance(f7d, pd.DataFrame):
            df = filter_sede(f7d)
            cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df.columns]
            st.dataframe(df[cols].head(48), width="stretch")

            if "timestamp" in df.columns and "yhat" in df.columns:
                tmp = df.copy()
                tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
                tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
                st.line_chart(tmp["yhat"].head(400))
        else:
            st.warning("No encuentro forecast_7d_energy.csv en /reports.")

# -----------------------------
# Anomalías y recomendaciones
# -----------------------------
with tab3:
    st.subheader("Anomalías y recomendaciones")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Anomalías (top)")
        if isinstance(anom, pd.DataFrame):
            df = filter_sede(anom)

            if show_only_anomalies and "is_anomaly" in df.columns:
                is_true = df["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])
                df = df[is_true]

            if "abs_residual" not in df.columns and "residual" in df.columns:
                df["abs_residual"] = df["residual"].abs()

            if "abs_residual" in df.columns:
                df = df.sort_values("abs_residual", ascending=False)

            cols = [
                c for c in
                ["timestamp", "sede_id", "sede", "energia_total_kwh", "yhat", "residual", "abs_residual", "threshold_p99"]
                if c in df.columns
            ]
            st.dataframe(df[cols].head(20), width="stretch")

            if len(df) == 0:
                st.info("No hay filas tras aplicar filtros (revisa sede o desmarca 'Solo anomalías').")
        else:
            st.warning("No encuentro anomalies_energy.csv en /reports.")

    with c4:
        st.markdown("### Recomendaciones (top)")
        if isinstance(reco, pd.DataFrame):
            df = filter_sede(reco)

            # Ordena si hay una columna de prioridad (severity suele existir)
            if "severity" in df.columns:
                df = df.sort_values("severity", ascending=False)

            st.dataframe(df.head(20), width="stretch")

            if len(df) == 0:
                st.info("No hay recomendaciones para el filtro actual.")
        else:
            st.warning("No encuentro recommendations_energy.csv en /reports.")

# -----------------------------
# Evidencia real vs pred
# -----------------------------
with tab4:
    st.subheader("Evidencia: real vs pred (periodo de test)")
    if isinstance(pv, pd.DataFrame):
        df = filter_sede(pv)
        cols = [c for c in ["timestamp", "sede_id", "sede", "y", "yhat"] if c in df.columns]
        st.dataframe(df[cols].head(80), width="stretch")

        if "timestamp" in df.columns and "y" in df.columns and "yhat" in df.columns:
            tmp = df.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp[["y", "yhat"]].head(400))
    else:
        st.warning("No encuentro pred_vs_real_energy.csv en /reports.")
