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
# Estilo
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
.block-container {
  max-width: 1150px;
  padding-top: 1.0rem;
  padding-bottom: 2.0rem;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,70,150,0.10) 0%, rgba(10,70,150,0.03) 60%, rgba(255,255,255,0) 100%);
  border-right: 1px solid rgba(0,0,0,0.08);
}
.header {
  border-radius: 16px;
  padding: 1rem 1.1rem;
  background: linear-gradient(90deg, rgba(10,70,150,0.95) 0%, rgba(0,160,190,0.85) 100%);
  color: white;
  margin-bottom: 0.9rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.header h1 { font-size: 1.6rem; margin: 0; letter-spacing: -0.02em; }
.header p { margin: 0.35rem 0 0 0; opacity: 0.95; }

.card {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.95rem;
  background: rgba(255,255,255,0.92);
  box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}
.kpi-title { font-size: 0.82rem; opacity: 0.70; margin-bottom: 0.15rem; }
.kpi-value { font-size: 1.35rem; font-weight: 800; margin: 0; }

.badge {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 800;
  margin-left: 0.35rem;
  border: 1px solid rgba(0,0,0,0.10);
}
.badge-alta { background: rgba(255, 59, 48, 0.12); color: rgba(150, 0, 0, 0.95); }
.badge-media { background: rgba(255, 149, 0, 0.14); color: rgba(140, 70, 0, 0.95); }
.badge-baja { background: rgba(52, 199, 89, 0.14); color: rgba(0, 100, 35, 0.95); }

.reco-title { font-weight: 900; margin: 0 0 0.35rem 0; }
.reco-text { margin: 0.15rem 0 0 0; line-height: 1.35; word-wrap: break-word; white-space: normal; }

div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.10);
}

hr { border: none; border-top: 1px solid rgba(0,0,0,0.10); margin: 1.2rem 0; }
</style>
        """.strip(),
        unsafe_allow_html=True,
    )


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


def badge_html(severity: str) -> str:
    sev = str(severity).strip().lower()
    if sev == "alta":
        return '<span class="badge badge-alta">ALTA</span>'
    if sev == "media":
        return '<span class="badge badge-media">MEDIA</span>'
    if sev == "baja":
        return '<span class="badge badge-baja">BAJA</span>'
    return '<span class="badge">N/A</span>'


# -----------------------------
# Recomendaciones (texto natural)
# -----------------------------
def build_reco_message(row: pd.Series) -> tuple[str, str]:
    sede = str(row.get("sede", "")).strip() or str(row.get("sede_id", "")).strip()
    ts_raw = row.get("timestamp")
    ts = pd.to_datetime(ts_raw, errors="coerce")

    y = row.get("y")
    yhat = row.get("yhat")
    abs_res = row.get("abs_residual")
    sev = str(row.get("severity", "")).strip().lower()

    es_fin = row.get("es_fin_semana", None)
    fuera_h = row.get("fuera_horario", None)

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

    ts_title = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else str(ts_raw)
    title = f"{sede} · {ts_title} {badge_html(sev)}"

    parts = [f"Se detectó un consumo anómalo en {sede} ({ctx_txt})."]
    if isinstance(y, (int, float)) and isinstance(yhat, (int, float)) and isinstance(abs_res, (int, float)):
        parts.append(f"Real: {y:.2f} kWh; esperado: {yhat:.2f} kWh; diferencia: {abs_res:.2f} kWh.")
    elif isinstance(abs_res, (int, float)):
        parts.append(f"Diferencia estimada: {abs_res:.2f} kWh.")

    action = str(row.get("action", "")).strip()
    if action:
        parts.append(f"Acción sugerida: {action}")

    return title, " ".join(parts)


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
# Agregaciones para comparativas
# -----------------------------
def top_sedes_24h(f24: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = f24.copy()
    if "yhat" not in df.columns or "sede_id" not in df.columns:
        return pd.DataFrame(columns=["sede_id", "kwh_24h"])
    out = df.groupby("sede_id", as_index=False)["yhat"].sum().rename(columns={"yhat": "kwh_24h"})
    out = out.sort_values("kwh_24h", ascending=False).head(top_n)
    return out


def anomalies_by_sede(anom: pd.DataFrame) -> pd.DataFrame:
    df = anom.copy()
    if "sede_id" not in df.columns:
        return pd.DataFrame(columns=["sede_id", "anomalies"])
    if "is_anomaly" in df.columns:
        is_true = df["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[is_true]
    out = df.groupby("sede_id", as_index=False).size().rename(columns={"size": "anomalies"})
    out = out.sort_values("anomalies", ascending=False)
    return out


def hourly_profile_24h(f24: pd.DataFrame, sedes: list[str]) -> pd.DataFrame:
    df = f24.copy()
    if "timestamp" not in df.columns or "yhat" not in df.columns or "sede_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["sede_id"].isin(sedes)]
    df["timestamp"] = safe_to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    agg = df.groupby(["hour", "sede_id"], as_index=False)["yhat"].mean()
    pivot = agg.pivot(index="hour", columns="sede_id", values="yhat").sort_index()
    return pivot


def daily_profile_7d(f7d: pd.DataFrame, sedes: list[str]) -> pd.DataFrame:
    df = f7d.copy()
    if "timestamp" not in df.columns or "yhat" not in df.columns or "sede_id" not in df.columns:
        return pd.DataFrame()
    df = df[df["sede_id"].isin(sedes)]
    df["timestamp"] = safe_to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    agg = df.groupby(["date", "sede_id"], as_index=False)["yhat"].sum()
    pivot = agg.pivot(index="date", columns="sede_id", values="yhat")
    return pivot


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="IAMinds UPTC · Energía", layout="centered")
inject_css()

st.markdown(
    """
<div class="header">
  <h1>IAMinds UPTC · Energía</h1>
  <p>Impacto · Comparativas · Pronóstico · Anomalías · Recomendaciones</p>
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

# Normaliza sede_id
for df in [f24, f7d, pv, anom, reco]:
    if isinstance(df, pd.DataFrame) and "sede_id" in df.columns:
        df["sede_id"] = normalize_sede_id(df["sede_id"])

# KPIs
target = metrics.get("target", "energia_total_kwh")
mae = metrics.get("mae")
rmse = metrics.get("rmse")
cutoff = metrics.get("cutoff_timestamp")

last_updated = metrics.get("generated_at")
if not last_updated and p_metrics:
    last_updated = pd.to_datetime(p_metrics.stat().st_mtime, unit="s").isoformat()

c1, c2 = st.columns(2)
with c1:
    kpi_card("Target", str(target))
with c2:
    kpi_card("Cutoff", str(cutoff))

c3, c4 = st.columns(2)
with c3:
    kpi_card("MAE", format_float(mae))
with c4:
    kpi_card("RMSE", format_float(rmse))

st.caption(f"Última actualización: {last_updated}")
st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# Sidebar: presets + filtros
# -----------------------------
st.sidebar.header("Acciones rápidas")

if "view" not in st.session_state:
    st.session_state["view"] = "home"

b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Inicio", use_container_width=True):
        st.session_state["view"] = "home"
with b2:
    if st.button("Comparar", use_container_width=True):
        st.session_state["view"] = "compare"

b3, b4 = st.sidebar.columns(2)
with b3:
    if st.button("Pronóstico", use_container_width=True):
        st.session_state["view"] = "forecast"
with b4:
    if st.button("Alertas", use_container_width=True):
        st.session_state["view"] = "alerts"

if st.sidebar.button("Recomendaciones", use_container_width=True):
    st.session_state["view"] = "reco"

if st.sidebar.button("Reset", use_container_width=True):
    for k in ["sede_id_sel", "compare_sedes", "only_anom", "view"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["view"] = "home"

st.sidebar.markdown("---")
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
sede_options = ["(todas)"] + (catalog["sede_id"].dropna().astype(str).unique().tolist() if len(catalog) else [])

if "sede_id_sel" not in st.session_state:
    st.session_state["sede_id_sel"] = "(todas)"

sede_id_sel = st.sidebar.selectbox("Sede", options=sede_options, key="sede_id_sel")

only_anom = st.sidebar.checkbox("Solo anomalías (is_anomaly=True)", value=True, key="only_anom")

# Para comparar: máximo 5 sedes, pero puedes elegir 2–5
default_compare = []
if len(catalog) >= 3:
    default_compare = catalog["sede_id"].astype(str).unique().tolist()[:3]
elif len(catalog) > 0:
    default_compare = catalog["sede_id"].astype(str).unique().tolist()

compare_sedes = st.sidebar.multiselect(
    "Comparar sedes (2–5)",
    options=catalog["sede_id"].astype(str).unique().tolist() if len(catalog) else [],
    default=default_compare,
    key="compare_sedes",
)

if len(compare_sedes) > 5:
    st.sidebar.warning("Máximo 5 sedes para comparar.")
    compare_sedes = compare_sedes[:5]
    st.session_state["compare_sedes"] = compare_sedes

def filter_sede(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sede_id_sel != "(todas)" and "sede_id" in out.columns:
        out = out[out["sede_id"] == sede_id_sel]
    return out


# -----------------------------
# Vistas (intuitivas por presets)
# -----------------------------
view = st.session_state.get("view", "home")

if view == "home":
    st.subheader("Impacto (vista rápida)")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Top sedes por consumo pronosticado (24h)**")
        if isinstance(f24, pd.DataFrame):
            top = top_sedes_24h(f24, top_n=10).set_index("sede_id")
            st.bar_chart(top)
        else:
            st.info("No está disponible forecast_24h_energy.csv")

    with colB:
        st.markdown("**Conteo de anomalías por sede**")
        if isinstance(anom, pd.DataFrame):
            ab = anomalies_by_sede(anom).head(10).set_index("sede_id")
            st.bar_chart(ab)
        else:
            st.info("No está disponible anomalies_energy.csv")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("**Curva 24h de la sede seleccionada**")
    if isinstance(f24, pd.DataFrame):
        df = filter_sede(f24)
        if sede_id_sel == "(todas)":
            st.info("Selecciona una sede para ver la curva.")
        else:
            if "timestamp" in df.columns and "yhat" in df.columns:
                tmp = df.copy()
                tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
                tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
                st.line_chart(tmp["yhat"].head(200))
            else:
                st.warning("El forecast 24h no tiene columnas esperadas.")
    else:
        st.info("No está disponible forecast_24h_energy.csv")

elif view == "compare":
    st.subheader("Comparar sedes")

    if not isinstance(f24, pd.DataFrame) or not isinstance(f7d, pd.DataFrame):
        st.warning("Necesitas forecast_24h_energy.csv y forecast_7d_energy.csv para comparar.")
    elif len(compare_sedes) < 2:
        st.info("Selecciona al menos 2 sedes en el panel lateral.")
    else:
        st.markdown("**Perfil por hora (promedio) – 24h**")
        hp = hourly_profile_24h(f24, compare_sedes)
        if hp.empty:
            st.info("No se pudo construir el perfil por hora.")
        else:
            st.line_chart(hp)

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("**Consumo diario (suma) – 7 días**")
        dp = daily_profile_7d(f7d, compare_sedes)
        if dp.empty:
            st.info("No se pudo construir el perfil por día.")
        else:
            st.line_chart(dp)

elif view == "forecast":
    st.subheader("Pronóstico")

    st.markdown("**24 horas**")
    if isinstance(f24, pd.DataFrame):
        df24 = filter_sede(f24)
        cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df24.columns]
        st.dataframe(df24[cols].head(60), width="stretch", hide_index=True)
        if sede_id_sel != "(todas)" and "timestamp" in df24.columns and "yhat" in df24.columns:
            tmp = df24.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp["yhat"].head(250))
    else:
        st.info("No está disponible forecast_24h_energy.csv")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("**7 días**")
    if isinstance(f7d, pd.DataFrame):
        df7 = filter_sede(f7d)
        cols = [c for c in ["timestamp", "sede_id", "sede", "yhat"] if c in df7.columns]
        st.dataframe(df7[cols].head(90), width="stretch", hide_index=True)
        if sede_id_sel != "(todas)" and "timestamp" in df7.columns and "yhat" in df7.columns:
            tmp = df7.copy()
            tmp["timestamp"] = safe_to_datetime(tmp["timestamp"])
            tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
            st.line_chart(tmp["yhat"].head(600))
    else:
        st.info("No está disponible forecast_7d_energy.csv")

elif view == "alerts":
    st.subheader("Anomalías")

    if isinstance(anom, pd.DataFrame):
        df = filter_sede(anom)

        if only_anom and "is_anomaly" in df.columns:
            is_true = df["is_anomaly"].astype(str).str.lower().isin(["true", "1", "yes"])
            df = df[is_true]

        if "abs_residual" not in df.columns and "residual" in df.columns:
            df["abs_residual"] = df["residual"].abs()

        if "abs_residual" in df.columns:
            df = df.sort_values("abs_residual", ascending=False)

        cols = [c for c in ["timestamp", "sede_id", "sede", "y", "yhat", "abs_residual", "threshold_p99"] if c in df.columns]
        st.dataframe(df[cols].head(60), width="stretch", hide_index=True)

        if len(df) == 0:
            st.info("No hay anomalías para el filtro actual.")
    else:
        st.info("No está disponible anomalies_energy.csv")

elif view == "reco":
    st.subheader("Recomendaciones")

    if isinstance(reco, pd.DataFrame):
        df = filter_sede(reco)

        # Orden por magnitud / severidad
        if "abs_residual" in df.columns:
            df = df.sort_values("abs_residual", ascending=False)
        elif "severity" in df.columns:
            order = {"alta": 0, "media": 1, "baja": 2}
            df["_sev_order"] = df["severity"].astype(str).str.lower().map(order).fillna(99)
            df = df.sort_values("_sev_order", ascending=True).drop(columns=["_sev_order"])

        limit = st.slider("Cantidad a mostrar", min_value=6, max_value=30, value=12, step=1)
        render_recommendations_cards(df, limit=limit)

        if len(df) == 0:
            st.info("No hay recomendaciones para el filtro actual.")

        with st.expander("Ver tabla (raw)"):
            cols = [
                c for c in
                ["timestamp", "sede_id", "sede", "y", "yhat", "abs_residual", "severity", "es_fin_semana", "fuera_horario", "action"]
                if c in df.columns
            ]
            st.dataframe(df[cols].head(80), width="stretch", hide_index=True)
    else:
        st.info("No está disponible recommendations_energy.csv")
else:
    st.session_state["view"] = "home"
