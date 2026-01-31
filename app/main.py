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
# Rutas y carga
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
    # Unifica UPTC-CHI y UPTC_CHI, además de casing/espacios
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace("-", "_", regex=False)
    return s


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def format_float(x: Any, ndigits: int = 4) -> str:
    if isinstance(x, (int, float)) and pd.notna(x):
        return f"{x:.{ndigits}f}"
    return str(x)


# -----------------------------
# Estilo (vertical + cards)
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
/* Contenedor más vertical (evita "pantalla enorme" y reduce scroll horizontal) */
.block-container {
  max-width: 1100px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}

/* Sidebar suave */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,70,150,0.10) 0%, rgba(10,70,150,0.03) 60%, rgba(255,255,255,0) 100%);
  border-right: 1px solid rgba(0,0,0,0.08);
}

/* Header */
.header {
  border-radius: 16px;
  padding: 1rem 1.1rem;
  background: linear-gradient(90deg, rgba(10,70,150,0.95) 0%, rgba(0,160,190,0.85) 100%);
  color: white;
  margin-bottom: 0.8rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.header h1 { font-size: 1.6rem; margin: 0; letter-spacing: -0.02em; }
.header p { margin: 0.35rem 0 0 0; opacity: 0.95; }

/* Cards base */
.card {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.95rem;
  background: rgba(255,255,255,0.92);
  box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

/* KPI */
.kpi-title { font-size: 0.82rem; opacity: 0.70; margin-bottom: 0.15rem; }
.kpi-value { font-size: 1.35rem; font-weight: 800; margin: 0; }

/* Badges severidad */
.badge {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  margin-left: 0.35rem;
  border: 1px solid rgba(0,0,0,0.10);
}
.badge-alta { background: rgba(255, 59, 48, 0.12); color: rgba(150, 0, 0, 0.95); }
.badge-media { background: rgba(255, 149, 0, 0.14); color: rgba(140, 70, 0, 0.95); }
.badge-baja { background: rgba(52, 199, 89, 0.14); color: rgba(0, 100, 35, 0.95); }

/* Reco card */
.reco-title { font-weight: 800; margin: 0 0 0.35rem 0; }
.reco-text { margin: 0.15rem 0 0 0; line-height: 1.35; word-wrap: break-word; white-space: normal; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* Dataframes: borde y radio */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.10);
}

/* Separador suave */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.10); margin: 1.2rem 0; }

</style>
        """.strip(),
        unsafe_allow_html=True,
    )


def badge_html(severity: str) -> str:
    sev = str(severity).strip().lower()
    if sev == "alta":
        return '<span class="badge badge-alta">ALTA</span>'
    if sev == "media":
        return '<span class="badge badge-media">MEDIA</span>'
    if sev == "baja":
        return '<span class="badge badge-baja">BAJA</span>'
    return '<span class="badge">N/A</span>'


def kpi_card(title: str, value: str) -> None:
    st.markdown(
        f"""
<div class="card">
  <div class="kpi-title">{title}</div>
  <p class="kpi-value">{value}</p>
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


# -----------------------------
# Recomendaciones en lenguaje natural (cards)
# -----------------------------
def build_reco_message(row: pd.Series) -> tuple[str, str]:
    """
    Devuelve (titulo, mensaje) en lenguaje natural, usando campos del CSV:
    timestamp, sede, y, yhat, abs_residual, severity, es_fin_semana, fuera_horario, action
    """
    sede = str(row.get("sede", "")).strip() or str(row.get("sede_id", "")).strip()
    ts_raw = row.get("timestamp")
    ts = pd.to_datetime(ts_raw, errors="coerce")

    y = row.get("y")
    yhat = row.get("yhat")
    abs_res = row.get("abs_residual")
    sev = str(row.get("severity", "")).strip().lower()

    es_fin = row.get("es_fin_semana", None)
    fuera_h = row.get("fuera_horario", None)

    # Contexto horario
    ctx = []
    if pd.notna(ts):
        hour = int(ts.hour)
        if 0 <= hour <= 5:
            ctx.append("madrugada")
        elif 6 <= hour <= 11:
            ctx.append("mañana")
        elif 12 <= hour <= 17:
            ctx.append("tarde")
        else:
            ctx.append("noche")

    if isinstance(fuera_h, (bool, int)) and bool(fuera_h):
        ctx.append("fuera de horario")
    if isinstance(es_fin, (bool, int)) and bool(es_fin):
        ctx.append("fin de semana")

    ctx_txt = ", ".join(ctx) if ctx else "evento"

    # Título corto
    ts_title = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else str(ts_raw)
    title = f"{sede} · {ts_title} {badge_html(sev)}"

    # Mensaje natural
    parts = []
    parts.append(f"Se detectó un consumo anómalo en {sede} ({ctx_txt}).")

    if isinstance(y, (int, float)) and isinstance(yhat, (int, float)) and isinstance(abs_res, (int, float)):
        parts.append(
            f"Consumo real: {y:.2f} kWh; esperado: {yhat:.2f} kWh; diferencia: {abs_res:.2f} kWh."
        )
    elif isinstance(abs_res, (int, float)):
        parts.append(f"Diferencia estimada: {abs_res:.2f} kWh.")

    action = str(row.get("action", "")).strip()
    if action:
        parts.append(f"Acción sugerida: {action}")

    msg = " ".join(parts)
    return title, msg


def render_recommendations_cards(df: pd.DataFrame, limit: int = 12) -> None:
    show = df.head(limit).copy()
    for _, r in show.iterrows():
        title, msg = build_reco_message(r)
        st.markdown(
            f"""
<div class="card" style="margin-bottom: 0.65rem;">
  <p class="reco-title">{title}</p>
  <p class="reco-text">{msg}</p>
</div>
            """.strip(),
            unsafe_allow_html=True,
        )


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="IAMinds UPTC - Energía", layout="centered")
inject_css()

st.markdown(
    """
<div class="header">
  <h1>IAMinds UPTC · Energía</h1>
  <p>Pronóstico por sede · Anomalías (p99) · Recomendaciones</p>
</div>
    """.strip(),
    unsafe_allow_html=True,
)

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

# Normalizar sede_id en todos los DF
for df in [f24, f7d, pv, anom, reco]:
    if isinstance(df, pd.DataFrame) and "sede_id" in df.columns:
        df["sede_id"] = normalize_sede_id(df["sede_id"])

# KPIs (vertical)
target = metrics.get("target", "energia_total_kwh")
mae = metrics.get("mae")
rmse = metrics.get("rmse")
cutoff = metrics.get("cutoff_timestamp")

split_method = metrics.get("split_method", "")
split_q: Optional[float] = None
if isinstance(split_method, str) and "quantile_" in split_method:
    try:
        split_q = float(split_method.split("quantile_")[-1])
    except ValueError:
        split_q = None

last_updated = metrics.get("generated_at")
if not last_updated and p_metrics:
    last_updated = pd.to_datetime(p_metrics.stat().st_mtime, unit="s").isoformat()

kcol1, kcol2 = st.columns(2)
with kcol1:
    kpi_card("Target", str(target))
with kcol2:
    kpi_card("Cutoff", str(cutoff))

kcol3, kcol4 = st.columns(2)
with kcol3:
    kpi_card("MAE", format_float(mae))
with kcol4:
    kpi_card("RMSE", format_float(rmse))

st.caption(f"Última actualización: {last_updated} · Directorio: {REPORTS_DIR.as_posix()}")
st.markdown("<hr/>", unsafe_allow_html=True)

# Sidebar filtros
st.sidebar.header("Filtros")

def build_catalog(*dfs: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for df in dfs:
        if not isinstance(df, pd.DataFrame) or "sede_id" not in df.columns:
            continue
        tmp = df[["sede_id"]].copy()
        tmp["sede"] = df["sede"].astype(str).str.strip() if "sede" in df.columns else ""
        rows.append(tmp.dropna(subset=["sede_id"]))
    if not rows:
        return pd.DataFrame(columns=["sede_id", "sede"])
    cat = pd.concat(rows, ignore_index=True).drop_duplicates()
    cat = cat.sort_values(["sede", "sede_id"], kind="stable")
    return cat


catalog = build_catalog(f24, f7d, pv, anom, reco)

label_to_id: dict[str, str] = {"(todas)": "(todas)"}
if len(catalog) > 0:
    for _, r in catalog.iterrows():
        sid = str(r["sede_id"])
        name = str(r.get("sede", "")).strip()
        label = f"{sid} — {name}" if name and name != "nan" else sid
        label_to_id[label] = sid
else:
    st.sidebar.warning("No se encontró 'sede_id' en los reportes.")

selected_label = st.sidebar.selectbox("Selecciona sede", options=list(label_to_id.keys()), key="sede_sel")
sede_id_sel = label_to_id[selected_label]

show_only_anomalies = st.sidebar.checkbox("Solo anomalías (is_anomaly=True)", value=True)

def filter_sede(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sede_id_sel != "(todas)" and "sede_id" in out.columns:
        out = out[out["sede_id"] == sede_id_sel]
    return out


# Tabs principales (vertical y claras)
tab1, tab2, tab3, tab4 = st.tabs(
    ["Resumen", "Pronósticos", "Anomalías", "Recomendaciones"]
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
    with cA:
        kpi_card("Filas forecast 24h", f"{len(f24):,}" if isinstance(f24, pd.DataFrame) else "NA")
    with cB:
        kpi_card("Filas forecast 7d", f"{len(f7d):,}" if isinstance(f7d, pd.DataFrame) else "NA")
    with cC:
        kpi_card("Filas evidencia test", f"{len(pv):,}" if isinstance(pv, pd.DataFrame) else "NA")

# -----------------------------
# Pronósticos
# -----------------------------
with tab2:
    st.subheader("Pronósticos por sede")

    if isinstance(f24, pd.DataFrame):
        st.markdown("#### Próximas 24 horas")
        df24 = filter_sede(f24)
        cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df24.columns]
        st.dataframe(df24[cols].head(48), width="stretch", hide_index=True)

        if "timestamp" in df24.columns and "yhat" in df24.columns:
            tmp = df24.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp["yhat"].head(200))
    else:
        st.warning("No existe forecast_24h_energy.csv en /reports.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    if isinstance(f7d, pd.DataFrame):
        st.markdown("#### Próximos 7 días")
        df7 = filter_sede(f7d)
        cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df7.columns]
        st.dataframe(df7[cols].head(72), width="stretch", hide_index=True)

        if "timestamp" in df7.columns and "yhat" in df7.columns:
            tmp = df7.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp["yhat"].head(500))
    else:
        st.warning("No existe forecast_7d_energy.csv en /reports.")

# -----------------------------
# Anomalías
# -----------------------------
with tab3:
    st.subheader("Anomalías detectadas")

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
            ["timestamp", "sede_id", "sede", "y", "yhat", "abs_residual", "threshold_p99"]
            if c in df.columns
        ]
        st.dataframe(df[cols].head(40), width="stretch", hide_index=True)

        if len(df) == 0:
            st.info("No hay anomalías para el filtro actual.")
    else:
        st.warning("No existe anomalies_energy.csv en /reports.")

# -----------------------------
# Recomendaciones (cards)
# -----------------------------
with tab4:
    st.subheader("Recomendaciones")

    if isinstance(reco, pd.DataFrame):
        df = filter_sede(reco)

        # Orden por severidad y magnitud (si existe abs_residual)
        if "abs_residual" in df.columns:
            df = df.sort_values("abs_residual", ascending=False)
        elif "severity" in df.columns:
            # fallback: severidad como string (alta > media > baja)
            order = {"alta": 0, "media": 1, "baja": 2}
            df["_sev_order"] = df["severity"].astype(str).str.lower().map(order).fillna(99)
            df = df.sort_values("_sev_order", ascending=True).drop(columns=["_sev_order"])

        limit = st.slider("Cantidad a mostrar", min_value=5, max_value=30, value=12, step=1)
        render_recommendations_cards(df, limit=limit)

        with st.expander("Ver tabla (raw)"):
            cols = [
                c for c in
                ["timestamp", "sede_id", "sede", "y", "yhat", "abs_residual", "severity", "es_fin_semana", "fuera_horario", "action"]
                if c in df.columns
            ]
            # Ojo: esta tabla puede generar scroll horizontal por el texto largo de action.
            # Por eso va en un expander opcional.
            st.dataframe(df[cols].head(50), width="stretch", hide_index=True)

        if len(df) == 0:
            st.info("No hay recomendaciones para el filtro actual.")
    else:
        st.warning("No existe recommendations_energy.csv en /reports.")
