# app.py — AseguraView con HistGradientBoostingRegressor (scikit-learn)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from io import BytesIO
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

# ====================== CONFIG UI ======================
st.set_page_config(
    page_title="AseguraView · Primas & Presupuesto",
    layout="wide",
    page_icon=":bar_chart:"
)

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
.badge{display:inline-block;position:relative;margin-left:6px}
.badge.right{margin-left:0;margin-right:6px}
.badge > .q{cursor:help;color:#93c5fd;font-weight:700}
.badge .tip{
  visibility:hidden;opacity:0;transition:opacity .15s;
  position:absolute;left:0;bottom:125%;width:320px;z-index:50;
  background:rgba(15,23,42,.96);color:#e5e7eb;border:1px solid rgba(148,163,184,.35);
  padding:10px 12px;border-radius:10px;font-size:12.5px;
}
.badge.right .tip{left:auto;right:0}
.badge:hover .tip{visibility:visible;opacity:1}
</style>
<div class="glow"></div>
""", unsafe_allow_html=True)

# Sonido de entrada
st.markdown("""
<audio id="intro-audio" src="https://cdn.pixabay.com/download/audio/2024/01/09/audio_ee3a8b2b42.mp3?filename=futuristic-digital-sweep-168473.mp3"></audio>
<script>
  const a = document.getElementById('intro-audio');
  if (a && !window._aseguraview_sound) { window._aseguraview_sound = true;
    setTimeout(()=>{ a.volume = 0.45; a.play().catch(()=>{}); }, 400); }
</script>
""", unsafe_allow_html=True)

# ====================== BRANDING ======================
LOGO_URL = "https://d7nxjt1whovz0.cloudfront.net/marketplace/logos/divisions/seguros-de-vida-del-estado.png"
HERO_URL = "https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=2400&auto=format&fit=crop"

st.markdown(f"""
<div class="glass" style="display:flex;align-items:center;gap:18px;margin-bottom:12px">
  <img src="{LOGO_URL}" alt="Seguros del Estado" style="height:48px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none'">
  <div>
    <div class="neon" style="font-size:20px;font-weight:700;">AseguraView · Colombiana Seguros del Estado S.A.</div>
    <div style="opacity:.75">Inteligencia de negocio en tiempo real · Forecast Gradient Boosting</div>
  </div>
</div>
<img src="{HERO_URL}" alt="hero" style="width:100%;height:180px;object-fit:cover;border-radius:18px;opacity:.35;margin-bottom:10px" onerror="this.style.display='none'">
""", unsafe_allow_html=True)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (Gradient Boosting), nowcast del mes en curso, cierre estimado 2025 y presupuesto sugerido 2026 (con IPC), por Año / Sucursal / Línea / Compañía.")

# ====================== DATOS ======================
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# -------- Helpers generales --------
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
    year_series = ts[ts.index.year == ref_year]
    if year_series.empty: return ts.dropna()
    mask = (year_series[::-1] == 0)
    run, flag = [], True
    for v in mask:
        if flag and bool(v): run.append(True)
        else: flag = False; run.append(False)
    trailing_zeros = pd.Series(run[::-1], index=year_series.index)
    ts.loc[trailing_zeros.index[trailing_zeros]] = np.nan
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

# -------- Features para el modelo --------
def _make_features_from_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    dfX = pd.DataFrame(index=idx)
    dfX["year"] = dfX.index.year.astype(np.int16)
    dfX["month"] = dfX.index.month.astype(np.int8)
    dummies = pd.get_dummies(dfX["month"], prefix="m")
    for m in range(1, 13):
        col = f"m_{m}"
        if col not in dummies.columns: dummies[col] = 0
    dummies = dummies[[f"m_{m}" for m in range(1, 13)]].astype(np.int8)
    dfX = pd.concat([dfX.drop(columns=["month"]), dummies], axis=1)
    dfX["t"] = np.arange(len(dfX), dtype=np.int32)
    return dfX

def _build_supervised(ts: pd.Series, lags=(1,2,3,6,12), rolls=(3,6,12)) -> pd.DataFrame:
    ts = ensure_monthly(ts.copy())
    df = ts.to_frame("y")
    if len(df) < (max(lags) + 6):
        return pd.DataFrame()
    for L in lags: df[f"lag_{L}"] = df["y"].shift(L)
    for W in rolls:
        rmean = df["y"].shift(1).rolling(W, min_periods=max(2, W//2)).mean()
        rstd  = df["y"].shift(1).rolling(W, min_periods=max(2, W//2)).std()
        df[f"roll_mean_{W}"] = rmean
        df[f"roll_std_{W}"]  = rstd
    df = df.join(_make_features_from_index(df.index))
    df = df.dropna()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df.astype(np.float32)

# -------- Modelo (HGBR) con fallback --------
def fit_xgb_forecast(ts_m: pd.Series, steps: int, eval_months:int=6):
    """
    Versión robusta con HistGradientBoostingRegressor (scikit-learn):
      - lags/rollings + dummies de mes + tendencia
      - validación temporal simple para SMAPE
      - PI por conformal (residuales absolutos del holdout)
      - fallback estacional si la muestra es corta
    """
    if steps < 1: steps = 1
    ts = ensure_monthly(ts_m.copy()).astype(np.float32)

    def _fallback(ts_local: pd.Series):
        future_idx = pd.date_range(ts_local.index.max() + pd.offsets.MonthBegin(), periods=steps, freq="MS")
        by_month = ts_local.groupby(ts_local.index.month).mean()
        preds = np.array([max(0.0, float(by_month.get(m, ts_local.mean()))) for m in future_idx.month], dtype=np.float32)
        hist_acum = ts_local.cumsum()
        forecast_acum = np.cumsum(preds) + (hist_acum.iloc[-1] if len(hist_acum)>0 else 0.0)
        fc_df = pd.DataFrame({
            "FECHA": future_idx,
            "Forecast_mensual": preds,
            "Forecast_acum": forecast_acum.astype(np.float32),
            "IC_lo": np.maximum(0, preds*0.85),
            "IC_hi": preds*1.15
        })
        hist_df = pd.DataFrame({"FECHA": ts_local.index, "Mensual": ts_local.values, "ACUM": hist_acum.values})
        return hist_df, fc_df, np.nan

    df_sup = _build_supervised(ts)
    if df_sup.empty or len(df_sup) < 24:
        return _fallback(ts)

    y = df_sup["y"].values.astype(np.float32)
    X = df_sup.drop(columns=["y"]).astype(np.float32)
    n = len(X)

    val_len = min(max(6, int(0.15*n)), n//3)
    if n - val_len < 12: val_len = max(3, n//5)

    X_tr, y_tr = X.iloc[:-val_len], y[:-val_len]
    X_va, y_va = X.iloc[-val_len:], y[-val_len:]

    base_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=None,
        max_iter=1000,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42
    )

    m_val = clone(base_model); m_val.fit(X_tr, y_tr)
    y_va_pred = m_val.predict(X_va)
    smape_val = smape(y_va, y_va_pred)
    resid_abs = np.abs(y_va - y_va_pred)
    q975 = float(np.quantile(resid_abs, 0.975)) if len(resid_abs) else 0.0

    model = clone(base_model); model.fit(X, y)
    feature_cols = list(X.columns)

    last_ts = ts.copy()
    future_idx = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=steps, freq="MS")
    preds, lo, hi = [], [], []

    for d in future_idx:
        tmp = last_ts.asfreq("MS")
        ext_idx = tmp.index.append(pd.DatetimeIndex([d]))
        tmp_ext = tmp.reindex(ext_idx)
        df_ext = _build_supervised(tmp_ext)
        if df_ext.empty:
            yhat = float((tmp.tail(12).mean() if len(tmp) else 0.0))
        else:
            X_pred_row = df_ext.iloc[[-1]].drop(columns=["y"], errors="ignore")
            X_pred_row = X_pred_row.reindex(columns=feature_cols, fill_value=0).astype(np.float32)
            yhat = float(model.predict(X_pred_row)[0])
        yhat = max(0.0, yhat)
        preds.append(yhat); lo.append(max(0.0, yhat - q975)); hi.append(yhat + q975)
        last_ts.loc[d] = yhat

    hist_acum = ts.cumsum()
    forecast_acum = np.cumsum(preds) + (hist_acum.iloc[-1] if len(hist_acum) else 0.0)
    fc_df = pd.DataFrame({
        "FECHA": future_idx,
        "Forecast_mensual": np.array(preds, dtype=np.float32),
        "Forecast_acum": np.array(forecast_acum, dtype=np.float32),
        "IC_lo": np.array(lo, dtype=np.float32),
        "IC_hi": np.array(hi, dtype=np.float32)
    })
    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values})
    return hist_df, fc_df, smape_val

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

# ====================== CARGA ======================
@st.cache_data(show_spinner=False)
def load_datos(url_csv: str) -> pd.DataFrame:
    df = pd.read_csv(url_csv)
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
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
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
    return df[keep].dropna(subset=['FECHA']).copy()

df = load_datos(gsheet_csv(SHEET_ID, SHEET_NAME_DATOS))

# ====================== FILTROS ======================
st.sidebar.header("Filtros")
years = sorted(df['ANIO'].dropna().unique())
year_sel = st.sidebar.multiselect("Año:", years, default=years,
                                  help="Filtra años a mostrar en tablas y gráficas históricas.")
suc_opts  = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts= ["TODAS"] + sorted(df['LINEA'].dropna().unique())    if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc  = st.sidebar.selectbox("Código y Sucursal:", suc_opts)
lin  = st.sidebar.selectbox("Línea:", linea_opts)
comp = st.sidebar.selectbox("Compañía:", comp_opts)

periodos_forecast = st.sidebar.number_input("Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1)

ipc_2026 = st.sidebar.number_input(
    "IPC proyectado para 2026 (%)", min_value=-5.0, max_value=30.0, value=7.0, step=0.1,
    help="Ajuste automático del presupuesto sugerido 2026."
)

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
    st.warning(f"Tu filtro no incluye el último año con datos ({ultimo_anio_datos}). Para faltantes/cierre se usa internamente {ultimo_anio_datos}.")

# ====================== TABS ======================
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
      • Motor <b>Gradient Boosting</b> con <i>lags</i>, ventanas móviles y estacionalidad (12 meses).<br>
      • Limpia ceros de cola y <b>excluye el mes actual si está parcial</b> (nowcast) para evitar sesgos a la baja.<br>
      • <b>Presupuesto 2026</b> ajustado por el <b>IPC proyectado</b> (<i>{ipc:.1f}%</i>).<br>
      • IC 95% por <i>conformal prediction</i>, tablas exportables y modo ejecutivo con escenarios.<br><br>
      <i>Nota:</i> Si el mes en curso no está completo (p. ej., septiembre hasta el día 23), se estima con el modelo (nowcast) y se suma al YTD.
    </div>
    """.format(ipc=ipc_2026), unsafe_allow_html=True)

# --------- TAB PRIMAS ---------
with tabs[1]:
    ref_year = int(df['FECHA'].max().year)

    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    serie_train, cur_month_ts, had_partial = split_series_excluding_partial_current(base_series, ref_year)

    if had_partial and cur_month_ts is not None:
        last_closed_month = cur_month_ts.month - 1
    else:
        last_closed_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_faltantes = max(0, 12 - last_closed_month)

    hist_df, fc_df, smape6 = fit_xgb_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6)

    nowcast_actual = None
    if had_partial and not fc_df.empty and cur_month_ts is not None:
        if fc_df.iloc[0]["FECHA"] != cur_month_ts:
            fc_df.iloc[0, fc_df.columns.get_loc("FECHA")] = cur_month_ts
        nowcast_actual = float(fc_df.iloc[0]["Forecast_mensual"])

    serie_2024 = ensure_monthly(serie_prima_all[serie_prima_all.index.year == 2024])
    df_2024 = pd.DataFrame({"FECHA": serie_2024.index, "Mensual_2024": serie_2024.values})
    df_2024["ACUM_2024"] = serie_2024.cumsum().values

    ytd_cerrado = serie_train[serie_train.index.year == ref_year].sum()
    ytd_ref = ytd_cerrado + (nowcast_actual if nowcast_actual is not None else 0.0)
    if had_partial and nowcast_actual is not None and len(fc_df) > 1:
        resto = fc_df['Forecast_mensual'].iloc[1:].sum()
    else:
        resto = fc_df['Forecast_mensual'].sum()
    cierre_ref = ytd_ref + resto

    cierre_2024 = float(serie_2024.sum()) if not serie_2024.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"YTD {ref_year}", fmt_cop(ytd_ref))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {ref_year}", fmt_cop(cierre_ref))
    c4.metric("Cierre anual 2024", fmt_cop(cierre_2024))

    st.markdown(f"#### Comparativo rápido: 2024 vs cierre estimado {ref_year}")
    comp_bar = pd.DataFrame({"Año": ["2024", f"Est. {ref_year}"], "Valor": [cierre_2024, cierre_ref]})
    fig_comp = px.bar(comp_bar, x="Año", y="Valor", text="Valor")
    fig_comp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_comp.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
    fig_comp.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="")
    fig_m = nicer_line(fig_m, "Primas mensuales (histórico) y forecast (Gradient Boosting)")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    if not df_2024.empty:
        fig_m.add_scatter(x=df_2024["FECHA"], y=df_2024["Mensual_2024"], name="2024 (mensual)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    st.plotly_chart(fig_m, use_container_width=True)

    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="")
    fig_a = nicer_line(fig_a, "Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    if not df_2024.empty:
        fig_a.add_scatter(x=df_2024["FECHA"], y=df_2024["ACUM_2024"], name="2024 (acum)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    if cierre_2024 > 0:
        fig_a.add_hline(y=cierre_2024, line_dash="dot", line_width=2,
                        annotation_text=f"Cierre 2024: {fmt_cop(cierre_2024)}",
                        annotation_position="top left")
    st.plotly_chart(fig_a, use_container_width=True)

    st.markdown(f"### Próximos meses proyectados (no cerrados en {ref_year})")
    if meses_faltantes > 0:
        meses_mostrar = st.slider(f"Meses a listar (faltantes de {ref_year}):", 1, meses_faltantes, min(6, meses_faltantes))
        sel = fc_df.head(meses_mostrar).copy()
        serie_2024_idx = serie_2024.copy()
        mismo_mes_2024 = []
        for d in sel["FECHA"]:
            try:
                valor = serie_2024_idx.loc[pd.Timestamp(year=2024, month=d.month, day=1)]
            except KeyError:
                valor = np.nan
            mismo_mes_2024.append(valor)
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            "Mismo mes 2024": np.array(mismo_mes_2024, dtype=float),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })
        show_df(tabla_faltantes, money_cols=["Mismo mes 2024","Proyección","IC 95% inf","IC 95% sup"], key="faltantes_2025")
    else:
        st.info(f"No quedan meses por cerrar en {ref_year} con los datos actuales.")

    st.success(f"Con nowcast para el mes en curso, el **YTD {ref_year}** es **{fmt_cop(ytd_ref)}** y el **cierre estimado** asciende a **{fmt_cop(cierre_ref)}**.")

    hist_tbl = hist_df.copy(); hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl   = fc_df.copy();   fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_bytes = to_excel_bytes({"Historico": hist_tbl, f"Forecast {ref_year} completo (GB)": fc_tbl})
    st.download_button("⬇️ Descargar Excel (PRIMAS)", data=xls_bytes,
                       file_name="primas_forecast_gradient_boosting.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[2]:
    st.subheader("Ejecución vs Presupuesto 2025 y Presupuesto sugerido 2026")
    st.caption(f"Nota: el presupuesto 2026 aplica un ajuste automático de **IPC proyectado {ipc_2026:.1f}%**.")

    ref_year = int(df['FECHA'].max().year)

    serie_exec = ensure_monthly(serie_prima_all)
    serie_pres = ensure_monthly(serie_presu_all)
    pres_ref = serie_pres[serie_pres.index.year == ref_year]

    serie_exec_clean0 = sanitize_trailing_zeros(serie_exec, ref_year)
    serie_exec_clean, cur_m_ref, had_partial_ref = split_series_excluding_partial_current(serie_exec_clean0, ref_year)

    if had_partial_ref and cur_m_ref is not None:
        last_closed_month_ref = cur_m_ref.month - 1
    else:
        last_closed_month_ref = last_actual_month_from_df(df_noYear, ref_year)
    meses_falt_ref = max(0, 12 - last_closed_month_ref)

    _, fc_ref, _ = fit_xgb_forecast(serie_exec_clean, steps=max(1, meses_falt_ref))

    nowcast_ref = None
    if had_partial_ref and not fc_ref.empty and cur_m_ref is not None:
        if fc_ref.iloc[0]["FECHA"] != cur_m_ref:
            fc_ref.iloc[0, fc_ref.columns.get_loc("FECHA")] = cur_m_ref
        nowcast_ref = float(fc_ref.iloc[0]["Forecast_mensual"])

    ytd_ejec_cerrado = serie_exec_clean[serie_exec_clean.index.year == ref_year].sum()
    ytd_ejec = ytd_ejec_cerrado + (nowcast_ref if nowcast_ref is not None else 0.0)

    this_month = pd.Timestamp.today().month
    ytd_pres = pres_ref.loc[pres_ref.index <= pd.Timestamp(f"{ref_year}-{this_month:02d}-01")].sum() if not pres_ref.empty else np.nan
    var_pct = ((ytd_ejec - ytd_pres) / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

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

    comp_ref = pd.DataFrame(index=pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS"))
    comp_ref["Presupuesto"] = pres_ref.reindex(comp_ref.index) if not pres_ref.empty else np.nan
    ejec_mes = serie_exec_clean.reindex(comp_ref.index)
    if had_partial_ref and cur_m_ref in comp_ref.index and nowcast_ref is not None:
        ejec_mes.loc[cur_m_ref] = nowcast_ref
    comp_ref["Ejecutado"] = ejec_mes

    if meses_falt_ref > 0 and not fc_ref.empty:
        fc_rest = fc_ref.copy()
        if had_partial_ref and cur_m_ref is not None and len(fc_rest) > 0:
            fc_rest = fc_rest.iloc[1:]
        if len(fc_rest) > 0:
            comp_ref.loc[fc_rest["FECHA"], "Proyección ejecución"] = fc_rest.set_index("FECHA")["Forecast_mensual"]

    figp = px.line(comp_ref.reset_index(names="FECHA"), x="FECHA",
                   y=[c for c in ["Presupuesto","Ejecutado","Proyección ejecución"] if c in comp_ref.columns], title="")
    figp = nicer_line(figp, f"{ref_year}: Presupuesto vs Ejecutado (con nowcast) y proyección")
    st.plotly_chart(figp, use_container_width=True)

    # Sugerido 2026 (Gradient Boosting) + IPC
    serie_exec_clean_local = serie_exec_clean.copy()
    if len(serie_exec_clean_local.dropna()) < 18:
        st.info("Muestra filtrada corta: usamos comportamiento global como referencia para 2026.")
        serie_exec_global = ensure_monthly(df.groupby('FECHA')['IMP_PRIMA'].sum().sort_index())
        serie_exec_clean_local = sanitize_trailing_zeros(serie_exec_global, ref_year)

    pasos_total = max(1, meses_falt_ref) + 12
    _, fc_ext, _ = fit_xgb_forecast(serie_exec_clean_local, steps=pasos_total, eval_months=6)
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
                       file_name="presupuesto_refyear_y_2026_ipc_gradient_boosting.xlsx",
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
            base_col = base_26.columns[2]  # Ajuste IPC XX.X%
            base_26["Escenario ajustado 2026"] = (base_26[base_col]*(1+ajuste_pct/100)).round(0).astype(int)
            total_base = int(base_26[base_col].sum())
            total_adj  = int(base_26["Escenario ajustado 2026"].sum())
            c1,c2 = st.columns(2)
            c1.metric("Total 2026 (con IPC)", fmt_cop(total_base))
            c2.metric("Total escenario 2026", fmt_cop(total_adj), delta=f"{ajuste_pct:+d}%")
            show_df(base_26[["FECHA",base_col,"Escenario ajustado 2026"]], key="escenario26")
            xls_dir = to_excel_bytes({"2026_con_IPC_vs_ajustado": base_26.assign(FECHA=base_26["FECHA"].dt.strftime("%Y-%m"))})
            st.download_button("⬇️ Descargar Excel (Modo Director - 2026)", data=xls_dir,
                               file_name="modo_director_2026_gradient_boosting.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with colB:
        st.markdown("#### Stress test / Tornado " + info_badge(
            "Compara 3 escenarios: Base (con IPC), -X% y +X%. Mide cuánto cambiaría el total del año."
        ), unsafe_allow_html=True)
        perc = st.select_slider("Rango de sensibilidad", options=[5,10,15,20,25,30], value=10)
        if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty:
            base_26 = presupuesto_2026_df.copy()
            base_col = base_26.columns[2]
            up = int((base_26[base_col]*(1+perc/100)).sum())
            dn = int((base_26[base_col]*(1-perc/100)).sum())
            bench = int(base_26[base_col].sum())
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
        base_series2 = sanitize_trailing_zeros(serie_prima_all.copy(), yref)
        serie_train2, cur_ts2, had_part2 = split_series_excluding_partial_current(base_series2, yref)
        falt2 = max(0, 12 - (cur_ts2.month - 1 if had_part2 and cur_ts2 is not None else last_actual_month_from_df(df_noYear, yref)))
        _, fc_tmp2, _ = fit_xgb_forecast(serie_train2, steps=max(1, falt2))
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
                f"Para **2026**, el **presupuesto (con IPC {ipc_2026:.1f}%)** asciende a **{fmt_cop(total_26)}**.")
    except:
        pass
