# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO

# ============ CONFIG ============
st.set_page_config(
    page_title="AseguraView · Primas & Presupuesto",
    layout="wide",
    page_icon=":bar_chart:"
)

# Tema base + estilos y tooltips sin JS
st.markdown("""
<style>
body {background: radial-gradient(1200px 800px at 10% 10%, #0b1220 0%, #0a0f1a 45%, #060a12 100%) !important;}
.stApp {background-color: #111827;}
.block-container { padding-top: 0.6rem; }
.stDataFrame th, .stDataFrame td {color: #d8e2ff !important;}
.glass {background: rgba(17, 24, 39, 0.35); border: 1px solid rgba(255,255,255,0.08);
       box-shadow: 0 10px 30px rgba(0,0,0,0.35); backdrop-filter: blur(10px);
       border-radius: 18px; padding: 14px 18px;}
.neon { text-shadow: 0 0 10px #3b82f6, 0 0 20px #3b82f6; }
.glow {position: fixed; inset: auto auto 20px 20px; width: 220px; height: 220px;
       background: radial-gradient(closest-side at 50% 50%, rgba(59,130,246,.28), rgba(59,130,246,0));
       filter: blur(20px); pointer-events: none; z-index: 0; opacity:.6;}
/* Tooltips (CSS-only) */
.badge{display:inline-block;position:relative;margin-left:6px}
.badge.right{margin-left:0;margin-right:6px}
.badge > .q{cursor:help;color:#93c5fd;font-weight:700}
.badge .tip{
  visibility:hidden;opacity:0;transition:opacity .15s;
  position:absolute;left:0;bottom:125%;width:320px;z-index:50;
  background:rgba(15,23,42,.96);color:#e5e7eb;border:1px solid rgba(148,163,184,.35);
  padding:10px 12px;border-radius:10px;font-size:12.5px;border-radius:10px;
}
.badge.right .tip{left:auto;right:0}
.badge:hover .tip{visibility:visible;opacity:1}
</style>
<div class="glow"></div>
""", unsafe_allow_html=True)

# ======= BRANDING =======
LOGO_URL = "https://d7nxjt1whovz0.cloudfront.net/marketplace/logos/divisions/seguros-de-vida-del-estado.png"
HERO_URL = "https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=2400&auto=format&fit=crop"

st.markdown(f"""
<div class="glass" style="display:flex;align-items:center;gap:18px;margin-bottom:12px">
  <img src="{LOGO_URL}" alt="Seguros del Estado" style="height:48px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none'">
  <div>
    <div class="neon" style="font-size:20px;font-weight:700;">AseguraView · Colombiana Seguros del Estado S.A.</div>
    <div style="opacity:.75">Inteligencia de negocio en tiempo real · Forecast SARIMAX / XGBoost</div>
  </div>
</div>
<img src="{HERO_URL}" alt="hero" style="width:100%;height:180px;object-fit:cover;border-radius:18px;opacity:.35;margin-bottom:10px" onerror="this.style.display='none'">
""", unsafe_allow_html=True)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (SARIMAX/XGBoost), nowcast del mes en curso, cierre estimado 2025 y presupuesto sugerido 2026 (con IPC), por Año / Sucursal / Línea / Compañía.")

# ============ DATOS ============
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"   # <-- cambia si usas otro
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# ============ HELPERS ============
def info_badge(texto:str, right:bool=False) -> str:
    cls = "badge right" if right else "badge"
    return f'<span class="{cls}"><span class="q">❓</span><span class="tip">{texto}</span></span>'

def parse_number_co(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monthly(ts: pd.Series) -> pd.Series:
    ts = ts.asfreq("MS")
    ts = ts.interpolate(method="linear", limit_area="inside")
    return ts

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float); y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def last_actual_month_from_df(df_like: pd.DataFrame, ref_year: int) -> int:
    d = df_like.copy(); d = d[d['FECHA'].dt.year == ref_year]
    if 'IMP_PRIMA' not in d.columns: return 0
    d = d[d['IMP_PRIMA'].fillna(0) > 0]
    if d.empty: return 0
    return int(d['FECHA'].max().month)

def sanitize_trailing_zeros(ts: pd.Series, ref_year: int) -> pd.Series:
    ts = ensure_monthly(ts.copy())
    idx = ts.index[(ts.index.year == ref_year)]
    if len(idx) == 0:
        return ts.dropna()
    vals = ts.loc[idx].copy()
    for d in sorted(idx, reverse=True):
        if pd.isna(vals.loc[d]) or vals.loc[d] != 0:
            break
        ts.loc[d] = np.nan
    if ts.last_valid_index() is not None:
        ts = ts.loc[:ts.last_valid_index()]
    return ts.dropna()

def split_series_excluding_partial_current(ts: pd.Series, ref_year: int):
    ts = ensure_monthly(ts.copy())
    today = pd.Timestamp.today()
    cur_m = pd.Timestamp(year=today.year, month=today.month, day=1)
    if len(ts) == 0:
        return ts, None, False
    if ts.index.max() == cur_m and today.day < 28:
        ts.loc[cur_m] = np.nan
        return ts.dropna(), cur_m, True
    return ts.dropna(), None, False

def to_excel_bytes(sheets: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

def fmt_cop(x):
    try: return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception: return x

def show_df(df, money_cols=None, index=False, key=None):
    if money_cols is None:
        money_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    d = df.copy()
    for c in money_cols: d[c] = d[c].map(fmt_cop)
    st.dataframe(d, use_container_width=True, hide_index=not index, key=key)

def nicer_line(fig: Figure, title: str):
    fig.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=2))
    fig.update_layout(
        title=title, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(rangeslider_visible=True)
    for tr in fig.data:
        tr.hovertemplate = "%{x|%b-%Y}<br>%{y:,.0f}<extra></extra>"
    return fig

# ======== MODELOS ========

def fit_forecast_sarimax(ts_m: pd.Series, steps: int, eval_months:int=6):
    if steps < 1: steps = 1
    ts = ensure_monthly(ts_m.copy()); y = np.log1p(ts)

    smapes = []
    start = max(len(y)-eval_months, 12)
    if len(y) >= start+1:
        for t in range(start, len(y)):
            y_tr = y.iloc[:t]; y_te = y.iloc[t:t+1]
            try:
                m = SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                p = r.get_forecast(steps=1).predicted_mean
            except Exception:
                r = ARIMA(y_tr, order=(1,1,1)).fit()
                p = r.get_forecast(steps=1).predicted_mean
            smapes.append(smape(np.expm1(y_te.values), np.expm1(p.values)))
    smape_last = np.mean(smapes) if smapes else np.nan

    try:
        m_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                         enforce_stationarity=False, enforce_invertibility=False)
        r_full = m_full.fit(disp=False)
        pred = r_full.get_forecast(steps=steps)
        mean = np.expm1(pred.predicted_mean); ci = np.expm1(pred.conf_int(alpha=0.05))
    except Exception:
        r_full = ARIMA(y, order=(1,1,1)).fit()
        pred = r_full.get_forecast(steps=steps)
        mean = np.expm1(pred.predicted_mean); ci = np.expm1(pred.conf_int(alpha=0.05))

    future_idx = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    hist_acum = ts.cumsum()
    forecast_acum = np.cumsum(mean) + hist_acum.iloc[-1]

    fc_df = pd.DataFrame({
        "FECHA": future_idx,
        "Forecast_mensual": mean.values,
        "Forecast_acum": forecast_acum.values,
        "IC_lo": ci.iloc[:,0].values if hasattr(ci, "iloc") else np.maximum(mean.values*0.7, 0),
        "IC_hi": ci.iloc[:,1].values if hasattr(ci, "iloc") else mean.values*1.3
    })
    fc_df["IC_lo"] = fc_df["IC_lo"].clip(lower=0)
    fc_df["Forecast_mensual"] = fc_df["Forecast_mensual"].clip(lower=0)

    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values})
    return hist_df, fc_df, smape_last

def _build_xgb_dataset(y: pd.Series, exog: pd.Series|None):
    """Crea features con lags/rolling + estacionalidad mensual + exógena (PRESUPUESTO)."""
    df = pd.DataFrame({"y": y})
    if exog is not None:
        df["pres"] = exog.reindex(y.index).astype(float)

    # Lags y rolling
    for L in [1,2,3,12]:
        df[f"lag{L}"] = df["y"].shift(L)
    for W in [3,6,12]:
        df[f"ma{W}"] = df["y"].rolling(W, min_periods=max(2, W//2)).mean()

    # Estacionalidad con sin/cos del mes
    m = y.index.month
    df["m_sin"] = np.sin(2*np.pi*m/12)
    df["m_cos"] = np.cos(2*np.pi*m/12)

    # Tendencia simple
    df["t"] = np.arange(len(df))

    # Target log1p
    df["y_log"] = np.log1p(df["y"].clip(lower=0))

    # Drop filas con NaN por lags
    df = df.dropna().copy()
    feat_cols = [c for c in df.columns if c not in ["y","y_log"]]
    X = df[feat_cols].astype(float)
    ylog = df["y_log"].astype(float)
    return X, ylog, feat_cols, df.index

def fit_forecast_xgb(serie: pd.Series, steps:int, exog_series: pd.Series|None, eval_months:int=6):
    """Modelo árbol de gradiente (XGBoost). Predicción iterativa mes a mes."""
    if steps < 1: steps = 1
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise RuntimeError("xgboost no disponible: " + str(e))

    y = ensure_monthly(serie.copy())
    exog = ensure_monthly(exog_series) if exog_series is not None else None

    # Backtesting sMAPE
    smapes=[]
    y_bt = y.copy()
    start = max(len(y_bt)-eval_months, 18)  # pedimos más historia por lags
    if len(y_bt) >= start+1:
        for t in range(start, len(y_bt)):
            y_tr = y_bt.iloc[:t]
            ex_tr = exog.iloc[:t] if exog is not None else None
            X_tr, ylog_tr, feat_cols, _ = _build_xgb_dataset(y_tr, ex_tr)
            if len(X_tr) < 12:
                break
            model = XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42
            )
            model.fit(X_tr, ylog_tr)

            # Predicción 1 paso
            # reconstruimos features para el punto t usando todo hasta t-1
            y_tmp = y_tr.copy()
            future_idx = y_tr.index.max() + pd.offsets.MonthBegin()
            # arma fila de features:
            row = {
                "pres": float(ex_tr.loc[future_idx]) if ex_tr is not None and future_idx in ex_tr.index else np.nan,
                "lag1": y_tmp.iloc[-1],
                "lag2": y_tmp.iloc[-2] if len(y_tmp)>=2 else y_tmp.iloc[-1],
                "lag3": y_tmp.iloc[-3] if len(y_tmp)>=3 else y_tmp.iloc[-1],
                "lag12": y_tmp.iloc[-12] if len(y_tmp)>=12 else y_tmp.mean(),
                "ma3": y_tmp.tail(3).mean(),
                "ma6": y_tmp.tail(6).mean(),
                "ma12": y_tmp.tail(12).mean(),
                "m_sin": np.sin(2*np.pi*future_idx.month/12),
                "m_cos": np.cos(2*np.pi*future_idx.month/12),
                "t": len(y_tmp)
            }
            X1 = pd.DataFrame([row])[feat_cols].astype(float)
            pred_log = float(model.predict(X1)[0])
            pred = float(np.expm1(pred_log))
            smapes.append(smape([y_bt.iloc[t]], [pred]))

    smape_last = np.mean(smapes) if smapes else np.nan

    # Entrenamiento final
    X_all, ylog_all, feat_cols, idx_all = _build_xgb_dataset(y, exog)
    if len(X_all) < 12:
        # muy pocos puntos después de construir lags -> usar SARIMAX
        return fit_forecast_sarimax(serie, steps, eval_months)

    model_full = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42
    )
    model_full.fit(X_all, ylog_all)

    # Predicción iterativa pasos futuros
    y_hist = y.copy()
    preds = []
    future_idx = []
    cur_idx = y_hist.index.max()
    for h in range(steps):
        fi = cur_idx + pd.offsets.MonthBegin()
        # construir features con la serie extendida (hist + pred)
        y_tmp = ensure_monthly(y_hist)
        row = {
            "pres": float(exog.loc[fi]) if exog is not None and fi in exog.index else np.nan,
            "lag1": y_tmp.iloc[-1],
            "lag2": y_tmp.iloc[-2] if len(y_tmp)>=2 else y_tmp.iloc[-1],
            "lag3": y_tmp.iloc[-3] if len(y_tmp)>=3 else y_tmp.iloc[-1],
            "lag12": y_tmp.iloc[-12] if len(y_tmp)>=12 else y_tmp.mean(),
            "ma3": y_tmp.tail(3).mean(),
            "ma6": y_tmp.tail(6).mean(),
            "ma12": y_tmp.tail(12).mean(),
            "m_sin": np.sin(2*np.pi*fi.month/12),
            "m_cos": np.cos(2*np.pi*fi.month/12),
            "t": len(y_tmp)
        }
        Xf = pd.DataFrame([row])[feat_cols].astype(float)
        pred_log = float(model_full.predict(Xf)[0])
        pred = float(np.expm1(pred_log))
        pred = max(pred, 0.0)
        preds.append(pred); future_idx.append(pd.Timestamp(fi))
        # extiende la serie para siguientes lags
        y_hist = pd.concat([y_hist, pd.Series([pred], index=[fi])])
        cur_idx = fi

    hist_acum = y.cumsum()
    forecast_acum = np.cumsum(preds) + (hist_acum.iloc[-1] if len(hist_acum)>0 else 0.0)

    fc_df = pd.DataFrame({
        "FECHA": future_idx,
        "Forecast_mensual": preds,
        "Forecast_acum": forecast_acum,
        # CI aproximada por ±20% (si deseas, reemplazar por bootstrap)
        "IC_lo": np.maximum(np.array(preds)*0.8, 0),
        "IC_hi": np.array(preds)*1.2
    })
    hist_df = pd.DataFrame({"FECHA": y.index, "Mensual": y.values, "ACUM": hist_acum.values})
    return hist_df, fc_df, smape_last

def fit_forecast_any(modelo:str, ts_m: pd.Series, steps:int, eval_months:int, exog_series: pd.Series|None=None, peso_hibrido:float=0.5):
    modelo = (modelo or "SARIMAX").upper()
    if modelo == "SARIMAX":
        return fit_forecast_sarimax(ts_m, steps, eval_months)
    elif modelo == "XGBOOST":
        try:
            return fit_forecast_xgb(ts_m, steps, exog_series, eval_months)
        except Exception as e:
            st.warning(f"No se pudo usar XGBoost ({e}). Se usa SARIMAX.")
            return fit_forecast_sarimax(ts_m, steps, eval_months)
    elif modelo == "HÍBRIDO" or modelo == "HIBRIDO":
        # Combinar medias de ambos modelos
        h1, f1, e1 = fit_forecast_sarimax(ts_m, steps, eval_months)
        try:
            h2, f2, e2 = fit_forecast_xgb(ts_m, steps, exog_series, eval_months)
        except Exception as e:
            st.warning(f"No se pudo usar XGBoost en el híbrido ({e}). Se usa SARIMAX.")
            return h1, f1, e1
        # Promedio ponderado
        w = float(np.clip(peso_hibrido, 0.0, 1.0))
        f = f1.copy()
        f["Forecast_mensual"] = (w*f2["Forecast_mensual"].values + (1-w)*f1["Forecast_mensual"].values)
        f["IC_lo"] = np.minimum(f1["IC_lo"].values, f2["IC_lo"].values)
        f["IC_hi"] = np.maximum(f1["IC_hi"].values, f2["IC_hi"].values)
        f["Forecast_acum"] = np.cumsum(f["Forecast_mensual"].values) + h1["ACUM"].iloc[-1]
        sm = np.nanmean([e for e in [e1,e2] if not np.isnan(e)]) if any([not np.isnan(e1), not np.isnan(e2)]) else np.nan
        return h1, f, sm
    else:
        return fit_forecast_sarimax(ts_m, steps, eval_months)

# ============ CARGA ============
@st.cache_data(show_spinner=False)
def load_datos(url_csv: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url_csv)
    except Exception as e:
        st.error("No pude leer la hoja de Google (¿archivo público? ¿ID/hoja correctos?). Detalle: "+str(e))
        st.stop()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'Año':'ANIO','ANO':'ANIO','YEAR':'ANIO',
        'Mes yyyy':'MES_TXT','MES YYYY':'MES_TXT','Mes':'MES_TXT','MES':'MES_TXT',
        'Codigo y Sucursal':'SUCURSAL','Código y Sucursal':'SUCURSAL',
        'Linea':'LINEA','Línea':'LINEA',
        'Compañía':'COMPANIA','COMPAÑÍA':'COMPANIA','COMPANIA':'COMPANIA',
        'Imp Prima':'IMP_PRIMA','Imp Prima Cuota':'IMP_PRIMA_CUOTA'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if 'MES_TXT' in df.columns:
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], errors='coerce', infer_datetime_format=True)
        if df['FECHA'].isna().any():
            try:
                df.loc[df['FECHA'].isna(),'FECHA'] = pd.to_datetime(df.loc[df['FECHA'].isna(),'MES_TXT'], format='%b %Y')
            except Exception:
                pass
    else:
        df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str)+"-01-01", errors='coerce')
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()

    if 'IMP_PRIMA' in df.columns: df['IMP_PRIMA'] = parse_number_co(df['IMP_PRIMA'])
    if 'IMP_PRIMA_CUOTA' in df.columns: df['IMP_PRIMA_CUOTA'] = parse_number_co(df['IMP_PRIMA_CUOTA'])
    else:
        st.error("Falta la columna 'Imp Prima Cuota' (PRESUPUESTO)."); st.stop()
    df['PRESUPUESTO'] = df['IMP_PRIMA_CUOTA']

    for c in ['SUCURSAL','LINEA','COMPANIA']:
        if c in df.columns: df[c] = df[c].astype(str).str.strip().str.upper()
    if 'ANIO' not in df.columns: df['ANIO'] = df['FECHA'].dt.year

    keep = [x for x in ['ANIO','FECHA','SUCURSAL','LINEA','COMPANIA','IMP_PRIMA','PRESUPUESTO'] if x in df.columns]
    if 'IMP_PRIMA' not in keep:
        st.error("Falta la columna 'Imp Prima' (ejecutado). Verifica el archivo."); st.stop()
    return df[keep].dropna(subset=['FECHA']).copy()

df = load_datos(gsheet_csv(SHEET_ID, SHEET_NAME_DATOS))

# ============ FILTROS ============
st.sidebar.header("Filtros")

# Selector de modelo
modelo_sel = st.sidebar.selectbox(
    "Modelo de proyección:",
    ["SARIMAX", "XGBoost", "Híbrido"],
    help="XGBoost usa lags/medias móviles/mes sin-cos y PRESUPUESTO como exógena. Híbrido promedia ambos."
)

peso_hibrido = 0.6 if modelo_sel == "Híbrido" else 0.0
if modelo_sel == "Híbrido":
    peso_hibrido = st.sidebar.slider("Peso XGBoost en híbrido", 0.0, 1.0, 0.6, 0.05)

years = sorted(df['ANIO'].dropna().unique())
year_sel = st.sidebar.multiselect("Año:", years, default=years,
                                  help="Filtra años a mostrar en tablas y gráficas históricas.")
suc_opts  = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts= ["TODAS"] + sorted(df['LINEA'].dropna().unique())    if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc  = st.sidebar.selectbox("Código y Sucursal:", suc_opts)
lin  = st.sidebar.selectbox("Línea:", linea_opts)
comp = st.sidebar.selectbox("Compañía:", comp_opts)

periodos_forecast = st.sidebar.number_input(
    "Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1
)

# IPC proyectado 2026
ipc_2026 = st.sidebar.number_input(
    "IPC proyectado para 2026 (%)", min_value=-5.0, max_value=30.0, value=7.0, step=0.1,
    help="Aumento esperado de IPC para ajustar el presupuesto sugerido 2026."
)

# Botón de recarga
if st.sidebar.button("🔄 Recargar datos"):
    load_datos.clear()
    st.experimental_rerun()

# Filtro global para vistas; población para forecast (sin restringir años)
df_sel = df[df['ANIO'].isin(year_sel)].copy()
if suc != "TODAS" and 'SUCURSAL' in df_sel.columns: df_sel = df_sel[df_sel['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_sel.columns: df_sel = df_sel[df_sel['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_sel.columns: df_sel = df_sel[df_sel['COMPANIA'] == comp]

df_noYear = df.copy()
if suc != "TODAS" and 'SUCURSAL' in df_noYear.columns: df_noYear = df_noYear[df_noYear['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_noYear.columns: df_noYear = df_noYear[df_noYear['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_noYear.columns: df_noYear = df_noYear[df_noYear['COMPANIA'] == comp]

serie_prima_all = df_noYear.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu_all = df_noYear.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()
if serie_prima_all.empty:
    st.warning("No hay datos de IMP_PRIMA con los filtros seleccionados."); st.stop()

ultimo_anio_datos = int(df['FECHA'].max().year)
if ultimo_anio_datos not in year_sel:
    st.warning(f"Tu filtro no incluye el último año con datos ({ultimo_anio_datos}). "
               f"Para faltantes/cierre se usa internamente {ultimo_anio_datos}.")

# ============ TABS ============
tabs = st.tabs(["🏠 Presentación", "📈 Primas (forecast & cierre)", "🧭 Presupuesto 2026", "🧠 Modo Director BI"])

# --------- TAB PRESENTACIÓN ---------
with tabs[0]:
    st.markdown("## Bienvenido a **AseguraView**")
    st.markdown("""
    <div class="glass" style="padding:18px; line-height:1.5;">
      <b>¿Qué es?</b><br>
      AseguraView es el tablero corporativo para visualizar <b>Primas</b> y comparar la <b>ejecución vs presupuesto</b>,
      además de pronosticar <b>cierres de año</b> y sugerir el <b>presupuesto 2026</b> con base en el comportamiento mensual histórico.
      <br><br>
      <b>¿Cómo lo hace?</b><br>
      • <b>SARIMAX</b> (estacionalidad 12) y/o <b>XGBoost</b> con lags/rolling/estacionalidad + exógena <b>PRESUPUESTO</b>.<br>
      • Limpieza de ceros finales y <b>exclusión del mes actual si está parcial</b> (nowcast) para evitar sesgos a la baja.<br>
      • <b>Presupuesto 2026</b>: se ajusta automáticamente por el <b>IPC proyectado</b> que defines en el panel lateral (<i>{ipc:.1f}%</i>).<br>
      • IC 95% aprox., tablas exportables y modo ejecutivo con escenarios.
      <br><br>
      <i>Tip:</i> si ves cierres muy bajos, prueba <b>XGBoost</b> o el <b>Híbrido</b> (mayor peso a XGBoost).
    </div>
    """.format(ipc=ipc_2026), unsafe_allow_html=True)

# --------- TAB PRIMAS ---------
with tabs[1]:
    ref_year = int(df['FECHA'].max().year)

    # 1) Prepara series y excluye mes actual parcial para entrenar
    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    serie_train, cur_month_ts, had_partial = split_series_excluding_partial_current(base_series, ref_year)

    # Exógena para XGB (PRESUPUESTO)
    exog_full = sanitize_trailing_zeros(serie_presu_all.copy(), ref_year) if 'PRESUPUESTO' in df_noYear.columns or True else None
    # Alinea a índice de prima
    exog_full = ensure_monthly(exog_full).reindex(ensure_monthly(serie_prima_all).index) if exog_full is not None else None
    if exog_full is not None:
        exog_train = exog_full.reindex(serie_train.index)
    else:
        exog_train = None

    # 2) Define meses faltantes según último mes cerrado
    if had_partial and cur_month_ts is not None:
        last_closed_month = cur_month_ts.month - 1
    else:
        last_closed_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_faltantes = max(0, 12 - last_closed_month)

    # 3) Entrena y pronostica con modelo seleccionado
    hist_df, fc_df, smape6 = fit_forecast_any(
        modelo_sel, serie_train, max(1, meses_faltantes), 6, exog_series=exog_train, peso_hibrido=peso_hibrido
    )

    # 4) Si hay mes parcial, el primer paso es el nowcast del mes actual
    nowcast_actual = None
    if had_partial and not fc_df.empty and cur_month_ts is not None:
        if fc_df.iloc[0]["FECHA"] != cur_month_ts:
            fc_df.iloc[0, fc_df.columns.get_loc("FECHA")] = cur_month_ts
        nowcast_actual = float(fc_df.iloc[0]["Forecast_mensual"])

    # Serie prev_year para comparación
    prev_year = ref_year - 1
    serie_prev = ensure_monthly(serie_prima_all[serie_prima_all.index.year == prev_year])
    df_prev = pd.DataFrame({"FECHA": serie_prev.index, f"Mensual_{prev_year}": serie_prev.values})
    df_prev[f"ACUM_{prev_year}"] = serie_prev.cumsum().values

    # KPIs con nowcast del mes actual (si aplica)
    ytd_cerrado = serie_train[serie_train.index.year == ref_year].sum()
    ytd_ref = ytd_cerrado + (nowcast_actual if nowcast_actual is not None else 0.0)

    if had_partial and nowcast_actual is not None and len(fc_df) > 1:
        resto = fc_df['Forecast_mensual'].iloc[1:].sum()
    else:
        resto = fc_df['Forecast_mensual'].sum()
    cierre_ref = ytd_ref + resto

    cierre_prev = float(serie_prev.sum()) if not serie_prev.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"YTD {ref_year}", fmt_cop(ytd_ref))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {ref_year}", fmt_cop(cierre_ref))
    c4.metric(f"Cierre anual {prev_year}", fmt_cop(cierre_prev))

    # Comparativo prev vs cierre estimado ref_year
    st.markdown(f"#### Comparativo rápido: {prev_year} vs cierre estimado {ref_year}")
    comp_bar = pd.DataFrame({"Año": [str(prev_year), f"Est. {ref_year}"], "Valor": [cierre_prev, cierre_ref]})
    fig_comp = px.bar(comp_bar, x="Año", y="Valor", text="Valor")
    fig_comp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_comp.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
    fig_comp.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Mensual
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="")
    fig_m = nicer_line(fig_m, f"Primas mensuales (histórico) y forecast · Modelo: {modelo_sel}")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    if not df_prev.empty:
        fig_m.add_scatter(x=df_prev["FECHA"], y=df_prev[f"Mensual_{prev_year}"], name=f"{prev_year} (mensual)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    st.plotly_chart(fig_m, use_container_width=True)

    # Acumulado
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="")
    fig_a = nicer_line(fig_a, "Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    if not df_prev.empty:
        fig_a.add_scatter(x=df_prev["FECHA"], y=df_prev[f"ACUM_{prev_year}"], name=f"{prev_year} (acum)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    if cierre_prev > 0:
        fig_a.add_hline(y=cierre_prev, line_dash="dot", line_width=2,
                        annotation_text=f"Cierre {prev_year}: {fmt_cop(cierre_prev)}",
                        annotation_position="top left")
    st.plotly_chart(fig_a, use_container_width=True)

    # Tabla de faltantes
    st.markdown(f"### Próximos meses proyectados (no cerrados en {ref_year})")
    if meses_faltantes > 0:
        meses_mostrar = st.slider(f"Meses a listar (faltantes de {ref_year}):", 1, meses_faltantes, min(6, meses_faltantes))
        sel = fc_df.head(meses_mostrar).copy()
        # Mismo mes prev_year
        serie_prev_idx = serie_prev.copy()
        mismo_mes_prev = []
        for d in sel["FECHA"]:
            try:
                valor = serie_prev_idx.loc[pd.Timestamp(year=prev_year, month=d.month, day=1)]
            except KeyError:
                valor = np.nan
            mismo_mes_prev.append(valor)
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            f"Mismo mes {prev_year}": np.array(mismo_mes_prev, dtype=float),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })
        show_df(tabla_faltantes, money_cols=[f"Mismo mes {prev_year}","Proyección","IC 95% inf","IC 95% sup"], key="faltantes_2025")
    else:
        st.info(f"No quedan meses por cerrar en {ref_year} con los datos actuales.")

    st.success(f"Con nowcast para el mes en curso, el **YTD {ref_year}** es **{fmt_cop(ytd_ref)}** y el **cierre estimado** asciende a **{fmt_cop(cierre_ref)}**.")

    # Excel
    hist_tbl = hist_df.copy(); hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl   = fc_df.copy();   fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_bytes = to_excel_bytes({"Historico": hist_tbl, f"Forecast {ref_year} completo": fc_tbl})
    st.download_button("⬇️ Descargar Excel (PRIMAS)", data=xls_bytes,
                       file_name="primas_forecast.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[2]:
    st.subheader("Ejecución vs Presupuesto 2025 y Presupuesto sugerido 2026")
    st.caption(f"Nota: el presupuesto 2026 aplica un ajuste automático de **IPC proyectado {ipc_2026:.1f}%**.")

    ref_year = int(df['FECHA'].max().year)

    # Series ejecutado y presupuesto (mensuales)
    serie_exec = ensure_monthly(serie_prima_all)
    serie_pres = ensure_monthly(serie_presu_all)
    pres_ref = serie_pres[serie_pres.index.year == ref_year]

    # Excluir mes actual parcial en ejecutado
    serie_exec_clean0 = sanitize_trailing_zeros(serie_exec, ref_year)
    serie_exec_clean, cur_m_ref, had_partial_ref = split_series_excluding_partial_current(serie_exec_clean0, ref_year)

    if had_partial_ref and cur_m_ref is not None:
        last_closed_month_ref = cur_m_ref.month - 1
    else:
        last_closed_month_ref = last_actual_month_from_df(df_noYear, ref_year)
    meses_falt_ref = max(0, 12 - last_closed_month_ref)

    # Pronóstico de ejecución para completar el año (usar mismo modelo/mezcla)
    _, fc_ref, _ = fit_forecast_any(
        modelo_sel, serie_exec_clean, max(1, meses_falt_ref), 6,
        exog_series=pres_ref.reindex(serie_exec_clean.index) if modelo_sel!="SARIMAX" else None,
        peso_hibrido=peso_hibrido
    )

    # Nowcast para el mes actual (si parcial)
    nowcast_ref = None
    if had_partial_ref and not fc_ref.empty and cur_m_ref is not None:
        if fc_ref.iloc[0]["FECHA"] != cur_m_ref:
            fc_ref.iloc[0, fc_ref.columns.get_loc("FECHA")] = cur_m_ref
        nowcast_ref = float(fc_ref.iloc[0]["Forecast_mensual"])

    # Ejecutado YTD = cerrados + nowcast de mes actual (si aplica)
    ytd_ejec_cerrado = serie_exec_clean[serie_exec_clean.index.year == ref_year].sum()
    ytd_ejec = ytd_ejec_cerrado + (nowcast_ref if nowcast_ref is not None else 0.0)

    # Presupuesto YTD con corte al mes actual del sistema
    this_month = pd.Timestamp.today().month
    ytd_pres = pres_ref.loc[pres_ref.index <= pd.Timestamp(f"{ref_year}-{this_month:02d}-01")].sum() if not pres_ref.empty else np.nan
    var_pct = ((ytd_ejec - ytd_pres) / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

    # Resto del año
    if had_partial_ref and nowcast_ref is not None and len(fc_ref) > 1:
        resto_ref = fc_ref['Forecast_mensual'].iloc[1:].sum()
    else:
        resto_ref = fc_ref['Forecast_mensual'].sum()
    cierre_ejec_ref = ytd_ejec + resto_ref

    c1,c2,c3 = st.columns(3)
    c1.metric(f"Presupuesto {ref_year} YTD", fmt_cop(ytd_pres) if not np.isnan(ytd_pres) else "s/datos")
    c2.metric(f"Ejecutado {ref_year} YTD (con nowcast)", fmt_cop(ytd_ejec),
              delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric(f"Cierre estimado {ref_year} (ejecución)", fmt_cop(cierre_ejec_ref))

    # Serie mensual comparativa
    comp_ref = pd.DataFrame(index=pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS"))
    comp_ref["Presupuesto"] = pres_ref.reindex(comp_ref.index) if not pres_ref.empty else np.nan
    ejec_mes = serie_exec_clean.reindex(comp_ref.index)  # meses cerrados
    if had_partial_ref and cur_m_ref in comp_ref.index and nowcast_ref is not None:
        ejec_mes.loc[cur_m_ref] = nowcast_ref
    comp_ref["Ejecutado"] = ejec_mes

    if meses_falt_ref > 0 and not fc_ref.empty:
        fc_rest = fc_ref.copy()
        if had_partial_ref and cur_m_ref is not None and len(fc_rest) > 0:
            fc_rest = fc_rest.iloc[1:]  # evitar duplicar mes actual
        if len(fc_rest) > 0:
            comp_ref.loc[fc_rest["FECHA"], "Proyección ejecución"] = fc_rest.set_index("FECHA")["Forecast_mensual"]

    figp = px.line(
        comp_ref.reset_index(names="FECHA"), x="FECHA",
        y=[c for c in ["Presupuesto","Ejecutado","Proyección ejecución"] if c in comp_ref.columns], title=""
    )
    figp = nicer_line(figp, f"{ref_year}: Presupuesto vs Ejecutado (con nowcast) y proyección · {modelo_sel}")
    st.plotly_chart(figp, use_container_width=True)

    # Sugerido 2026 (modelo) + ajuste IPC
    serie_exec_clean_local = serie_exec_clean.copy()
    if len(serie_exec_clean_local.dropna()) < 18:
        st.info("Muestra filtrada corta: usamos comportamiento global como referencia para 2026.")
        serie_exec_global = ensure_monthly(df.groupby('FECHA')['IMP_PRIMA'].sum().sort_index())
        serie_exec_clean_local = sanitize_trailing_zeros(serie_exec_global, ref_year)

    pasos_total = max(1, meses_falt_ref) + 12
    _, fc_ext, _ = fit_forecast_any(
        modelo_sel, serie_exec_clean_local, pasos_total, 6,
        exog_series=serie_pres.reindex(serie_exec_clean_local.index) if modelo_sel!="SARIMAX" else None,
        peso_hibrido=peso_hibrido
    )
    sug_2026 = fc_ext.tail(12).set_index("FECHA"); sug_2026.index = pd.date_range("2026-01-01","2026-12-01",freq="MS")

    base_2026 = sug_2026["Forecast_mensual"].round(0).astype(int)
    ipc_factor = 1 + (ipc_2026/100.0)
    ajustado_2026 = (base_2026 * ipc_factor).round(0).astype(int)

    presupuesto_2026_df = pd.DataFrame({
        "FECHA": base_2026.index,
        "Sugerido modelo 2026": base_2026.values,
        f"Ajuste IPC {ipc_2026:.1f}%": ajustado_2026.values,
        "IC 95% inf": sug_2026["IC_lo"].round(0).astype(int).values,
        "IC 95% sup": sug_2026["IC_hi"].round(0).astype(int).values
    })
    total_base = int(base_2026.sum())
    total_ajust = int(ajustado_2026.sum())
    st.success(f"**Presupuesto 2026** — Base modelo: {fmt_cop(total_base)} · Con IPC {ipc_2026:.1f}%: **{fmt_cop(total_ajust)}**")

    show_df(presupuesto_2026_df, money_cols=["Sugerido modelo 2026", f"Ajuste IPC {ipc_2026:.1f}%", "IC 95% inf","IC 95% sup"], key="pres_2026")

    comp_ref_tbl = comp_ref.reset_index().rename(columns={"index":"FECHA"}); comp_ref_tbl["FECHA"] = comp_ref_tbl["FECHA"].dt.strftime("%Y-%m")
    p2026_tbl = presupuesto_2026_df.copy(); p2026_tbl["FECHA"] = p2026_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_pres = to_excel_bytes({f"{ref_year} Pres vs Ejec (nowcast)": comp_ref_tbl, f"2026 Presupuesto (IPC {ipc_2026:.1f}%)": p2026_tbl})
    st.download_button("⬇️ Descargar Excel (PRESUPUESTO)", data=xls_pres,
                       file_name="presupuesto_refyear_y_2026_ipc.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB MODO DIRECTOR BI ---------
with tabs[3]:
    st.subheader("Panel Ejecutivo · Sensibilidades & Hallazgos")
    colA,colB = st.columns([1,1])

    with colA:
        st.markdown("#### Escenario 2026 ajustado " + info_badge(
            "Mueve el porcentaje para ver cómo cambia el total anual si todos los meses de 2026 suben o bajan."
        ), unsafe_allow_html=True)
        ajuste_pct = st.slider("Ajuste vs. 2026 (con IPC) (±30%)", -30, 30, 0, 1)
        if 'presupuesto_2026_df' not in locals() or presupuesto_2026_df.empty:
            st.info("Primero calcula el 2026 (con IPC) en la pestaña anterior.")
        else:
            base_26 = presupuesto_2026_df.copy()
            col_ipc = f"Ajuste IPC {ipc_2026:.1f}%"
            base_26["Escenario ajustado 2026"] = (base_26[col_ipc]*(1+ajuste_pct/100)).round(0).astype(int)
            total_base = int(base_26[col_ipc].sum())
            total_adj  = int(base_26["Escenario ajustado 2026"].sum())
            c1,c2 = st.columns(2)
            c1.metric("Total 2026 (con IPC)", fmt_cop(total_base))
            c2.metric("Total escenario 2026", fmt_cop(total_adj), delta=f"{ajuste_pct:+d}%")
            show_df(base_26[["FECHA",col_ipc,"Escenario ajustado 2026"]], key="escenario26")
            xls_dir = to_excel_bytes({"2026_con_IPC_vs_ajustado": base_26.assign(FECHA=base_26["FECHA"].dt.strftime("%Y-%m"))})
            st.download_button("⬇️ Descargar Excel (Modo Director - 2026)", data=xls_dir,
                               file_name="modo_director_2026.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with colB:
        st.markdown("#### Stress test / Tornado " + info_badge(
            "Compara 3 escenarios: Base (con IPC), -X% y +X%. Mide cuánto cambiaría el total del año."
        ), unsafe_allow_html=True)
        perc = st.select_slider("Rango de sensibilidad", options=[5,10,15,20,25,30], value=10)
        if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty:
            base_26 = presupuesto_2026_df.copy()
            col_ipc = f"Ajuste IPC {ipc_2026:.1f}%"
            up = int((base_26[col_ipc]*(1+perc/100)).sum())
            dn = int((base_26[col_ipc]*(1-perc/100)).sum())
            bench = int(base_26[col_ipc].sum())
            tornado = pd.DataFrame({"Escenario":[f"-{perc}%", "Base", f"+{perc}%"], "Total":[dn, bench, up]})
            fig_t = px.bar(tornado, x="Escenario", y="Total", text="Total")
            fig_t.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_t.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10,r=10,t=20,b=20))
            fig_t.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Calcula primero el 2026 con IPC para ver la sensibilidad.")

    st.markdown("---")
    st.markdown("#### Hallazgos automáticos (anomalías)")
    st.caption("Detección por **z-score ≥ 2.5** sobre la serie mensual suavizada." +
               info_badge("Marcamos picos/caídas inusuales para explicar campañas, eventos o ajustes."), unsafe_allow_html=True)
    try:
        s = ensure_monthly(serie_prima_all).copy()
        if len(s) >= 24:
            ma = s.rolling(12, min_periods=6).mean()
            resid = (s - ma) / (s.rolling(12, min_periods=6).std() + 1e-9)
            outliers = resid[np.abs(resid) >= 2.5].dropna()
            if not outliers.empty:
                alert = pd.DataFrame({
                    "Fecha": outliers.index.strftime("%b-%Y"),
                    "Valor": s.loc[outliers.index].astype(int).values,
                    "Desviación": outliers.round(2).values
                }).sort_index()
                show_df(alert, money_cols=["Valor"], key="anomalias")
            else:
                st.success("No se detectaron anomalías significativas con z-score ≥ 2.5.")
        else:
            st.info("Se requieren ≥24 meses para análisis de anomalías.")
    except Exception as e:
        st.info(f"No se pudo calcular anomalías: {e}")

    st.markdown("---")
    try:
        yref = int(df['FECHA'].max().year)
        # Recalcular cierre ref con nowcast coherente usando el modelo seleccionado
        base_series2 = sanitize_trailing_zeros(serie_prima_all.copy(), yref)
        serie_train2, cur_ts2, had_part2 = split_series_excluding_partial_current(base_series2, yref)
        falt2 = max(0, 12 - (cur_ts2.month - 1 if had_part2 and cur_ts2 is not None else last_actual_month_from_df(df_noYear, yref)))
        _, fc_tmp2, _ = fit_forecast_any(
            modelo_sel, serie_train2, max(1, falt2), 6,
            exog_series=serie_presu_all.reindex(serie_train2.index) if modelo_sel!="SARIMAX" else None,
            peso_hibrido=peso_hibrido
        )
        if had_part2 and cur_ts2 is not None and len(fc_tmp2)>0:
            if fc_tmp2.iloc[0]["FECHA"] != cur_ts2:
                fc_tmp2.iloc[0, fc_tmp2.columns.get_loc("FECHA")] = cur_ts2
            now2 = float(fc_tmp2.iloc[0]["Forecast_mensual"])
            resto2 = fc_tmp2['Forecast_mensual'].iloc[1:].sum() if len(fc_tmp2)>1 else 0.0
        else:
            now2 = 0.0
            resto2 = fc_tmp2['Forecast_mensual'].sum()
        ytd_cerr2 = serie_train2[serie_train2.index.year == yref].sum()
        cierre_ref2 = int(ytd_cerr2 + now2 + resto2)

        total_26 = int(presupuesto_2026_df[presupuesto_2026_df.columns[2]].sum()) if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty else 0
        st.info(f"**Resumen ejecutivo** — Con nowcast del mes en curso, el **cierre {yref}** se estima en **{fmt_cop(cierre_ref2)}**. "
                f"Para **2026**, el **presupuesto (con IPC {ipc_2026:.1f}%)** asciende a **{fmt_cop(total_26)}**. · Modelo: {modelo_sel}")
    except:
        pass
