import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
body {background: #131927 !important; color: #fff;}
.stApp {background-color: #131927;}
.block-container { padding-top: 0.6rem; }
.stDataFrame th, .stDataFrame td {color: #e8f0fd !important;}
.kpi-row {display:flex;justify-content:space-between;margin-bottom:18px;}
.kpi-block {flex:1;padding:0 16px;}
.kpi-title{font-size:17px;font-weight:500;}
.kpi-value{font-size:2em;font-weight:700;}
.badge{display:inline-block;position:relative;margin-left:6px}
.badge .q{cursor:help;color:#93c5fd;font-weight:700}
.badge .tip{
  visibility:hidden;opacity:0;transition:opacity .15s;
  position:absolute;left:0;bottom:125%;width:320px;z-index:50;
  background:rgba(15,23,42,.96);color:#e5e7eb;border:1px solid rgba(148,163,184,.35);
  padding:10px 12px;border-radius:10px;font-size:12.5px;
}
.badge:hover .tip{visibility:visible;opacity:1}
.tabsbar {background:#161c2a;padding:7px 0 0 0;display:flex;align-items:center;gap:22px;margin-bottom:20px;border-bottom:2px solid #232b44;}
.tabsbar .tab {font-size:16px;font-weight:500;color:#dee5f7;display:flex;align-items:center;gap:8px;opacity:.85;}
.tabsbar .tab.active {color:#e43a7a;border-bottom:3px solid #e43a7a;padding-bottom:6px;}
.tabsbar .tab i {font-size:17px;}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://d7nxjt1whovz0.cloudfront.net/marketplace/logos/divisions/seguros-de-vida-del-estado.png"
HERO_URL = "https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=2400&auto=format&fit=crop"
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;margin-bottom:12px">
  <img src="{LOGO_URL}" alt="Seguros del Estado" style="height:48px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none'">
  <div>
    <div style="font-size:26px;font-weight:700;color:#e8f0fd;">AseguraView · Primas & Presupuesto</div>
    <div style="opacity:.75;color:#dee5f7;font-size:15px;">Forecast mensual (SARIMAX), nowcast del mes en curso, cierre estimado 2025 y presupuesto sugerido 2026 (con IPC), por Sucursal / Línea / Compañía.</div>
  </div>
</div>
<div style="width:100%;height:8px;"></div>
<div class="tabsbar">
  <span class="tab active"><i>🏠</i>Presentación</span>
  <span class="tab"><i>📈</i>Primas (forecast & cierre)</span>
  <span class="tab"><i>🧭</i>Presupuesto 2026</span>
  <span class="tab"><i>🧠</i>Modo Director BI</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Presentación", "Primas (forecast & cierre)", "Presupuesto 2026", "Modo Director BI"])

SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

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

def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6):
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
        "IC_lo": ci.iloc[:,0].values,
        "IC_hi": ci.iloc[:,1].values
    })
    fc_df["IC_lo"] = fc_df["IC_lo"].clip(lower=0)
    fc_df["Forecast_mensual"] = fc_df["Forecast_mensual"].clip(lower=0)

    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values})
    return hist_df, fc_df, smape_last

def fmt_cop(x):
    try: return "$" + f"{int(round(float(x))):,}".replace(",", ".")
    except Exception: return x

def show_df(df, money_cols=None, index=False, key=None):
    if money_cols is None:
        money_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    d = df.copy()
    for c in money_cols: d[c] = d[c].map(fmt_cop)
    st.dataframe(d, use_container_width=True, hide_index=not index, key=key)

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

# ============ FILTROS ============
st.sidebar.header("Filtros")
suc_opts  = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts= ["TODAS"] + sorted(df['LINEA'].dropna().unique())    if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc  = st.sidebar.selectbox("Código y Sucursal:", suc_opts)
lin  = st.sidebar.selectbox("Línea:", linea_opts)
comp = st.sidebar.selectbox("Compañía:", comp_opts)

periodos_forecast = st.sidebar.number_input(
    "Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1
)

ipc_2026 = st.sidebar.number_input(
    "IPC proyectado para 2026 (%)", min_value=-5.0, max_value=30.0, value=7.0, step=0.1,
    help="Aumento esperado de IPC para ajustar el presupuesto sugerido 2026."
)

df_sel = df.copy()
if suc != "TODAS" and 'SUCURSAL' in df_sel.columns: df_sel = df_sel[df_sel['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_sel.columns: df_sel = df_sel[df_sel['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_sel.columns: df_sel = df_sel[df_sel['COMPANIA'] == comp]

df_noYear = df_sel.copy()
serie_prima_all = df_noYear.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu_all = df_noYear.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()
if serie_prima_all.empty:
    st.warning("No hay datos de IMP_PRIMA con los filtros seleccionados."); st.stop()
ultimo_anio_datos = int(df['FECHA'].max().year)

with tabs[1]:
    ref_year = int(df['FECHA'].max().year)
    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    serie_train, cur_month_ts, had_partial = split_series_excluding_partial_current(base_series, ref_year)
    last_closed_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_diciembre = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS")
    meses_faltantes = max(0, 12 - last_closed_month)
    hist_df, fc_df, smape6 = fit_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6)

    # === KPIs superiores, como en la foto ===
    prod_2025 = serie_train[serie_train.index.year == ref_year].sum()
    cierre_ref = prod_2025 + fc_df["Forecast_mensual"].sum() if not fc_df.empty else prod_2025
    cierre_2024 = serie_prima_all[serie_prima_all.index.year == 2024].sum()
    smape_val = smape6 if not np.isnan(smape6) else 0.0

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-block">
        <div class="kpi-title">Producción 2025 <span class="badge"><span class="q">?</span><span class="tip">Corresponde a la suma de meses cerrados (sin forecast).</span></span></div>
        <div class="kpi-value">{fmt_cop(prod_2025)}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">Cierre estimado 2025</div>
        <div class="kpi-value">{fmt_cop(cierre_ref)}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">Cierre anual 2024</div>
        <div class="kpi-value">{fmt_cop(cierre_2024)}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">SMAPE validación <span class="badge"><span class="q">?</span><span class="tip">Error porcentual medio simétrico en la validación rolling de 6 meses.</span></span></div>
        <div class="kpi-value">{smape_val:.2f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TABLA GENERAL DE PROYECCIÓN MENSUAL ---
    tabla_general_valores = []
    for mes in meses_diciembre:
        real_value = serie_prima_all.get(mes, None)
        if pd.notnull(real_value) and real_value > 0:
            tabla_general_valores.append(real_value)
        else:
            idx_fc = None
            for j, row in fc_df.iterrows():
                if row["FECHA"].month == mes.month and row["FECHA"].year == mes.year:
                    idx_fc = j
                    break
            if idx_fc is not None:
                tabla_general_valores.append(fc_df.iloc[idx_fc]["Forecast_mensual"])
            else:
                tabla_general_valores.append(None)
    tabla_general = pd.DataFrame({
        "Mes": meses_diciembre.strftime("%b-%Y"),
        "Valor": tabla_general_valores
    })
    show_df(tabla_general, money_cols=["Valor"], key="tabla_general_mes")

    # --- TABLA DE PROYECCIÓN POR LÍNEA ---
    lineas = sorted(df_noYear["LINEA"].unique())
    tabla_lineas = []
    for mes in meses_diciembre:
        fila = {"Mes": mes.strftime("%b-%Y")}
        for linea in lineas:
            v = df_noYear[(df_noYear['FECHA'] == mes) & (df_noYear['LINEA'] == linea)]["IMP_PRIMA"].sum()
            if v > 0:
                fila[linea] = v
            else:
                idx_fc = None
                for j, row in fc_df.iterrows():
                    if row["FECHA"].month == mes.month and row["FECHA"].year == mes.year:
                        idx_fc = j
                        break
                if idx_fc is not None:
                    prop_df = df_noYear[(df_noYear['FECHA'] >= mes - pd.DateOffset(months=11)) & (df_noYear['FECHA'] <= mes)]
                    suma_linea = prop_df.groupby("LINEA")["IMP_PRIMA"].sum()
                    suma_total = suma_linea.sum()
                    prop = suma_linea / suma_total if suma_total > 0 else pd.Series([1/len(lineas)]*len(lineas), index=lineas)
                    fila[linea] = int(fc_df.iloc[idx_fc]["Forecast_mensual"] * prop.get(linea, 1/len(lineas)))
                else:
                    fila[linea] = None
        tabla_lineas.append(fila)
    df_lineas_tabla = pd.DataFrame(tabla_lineas)
    df_lineas_tabla = df_lineas_tabla[["Mes"] + lineas]
    show_df(df_lineas_tabla, money_cols=lineas, key="tabla_linea_mes")

with tabs[2]:
    st.markdown("""
    <h2 style="color:#e8f0fd;">Ejecución vs Presupuesto año actual y Presupuesto sugerido 2026</h2>
    <div style="opacity:.85;font-size:16px;">Nota: el presupuesto 2026 aplica un ajuste automático de <i>IPC proyectado {0:.1f}%</i>.</div>
    """.format(ipc_2026), unsafe_allow_html=True)

    ref_year = int(df['FECHA'].max().year)
    pres_2025 = serie_presu_all[serie_presu_all.index.year == ref_year].sum()
    cierre_2025 = serie_prima_all[serie_prima_all.index.year == ref_year].sum()
    pasos_total = 12
    _, fc_ext, _ = fit_forecast(serie_prima_all, steps=pasos_total, eval_months=6)
    meses_2026 = pd.date_range("2026-01-01","2026-12-01",freq="MS")
    sug_2026 = fc_ext.head(12).set_index("FECHA"); sug_2026.index = meses_2026
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

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-block">
        <div class="kpi-title">Presupuesto 2025</div>
        <div class="kpi-value">{fmt_cop(pres_2025)}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">Cierre ejecutado 2025</div>
        <div class="kpi-value">{fmt_cop(cierre_2025)}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">Presupuesto sugerido 2026</div>
        <div class="kpi-value">{fmt_cop(base_2026.sum())}</div>
      </div>
      <div class="kpi-block">
        <div class="kpi-title">Presupuesto 2026 con IPC</div>
        <div class="kpi-value">{fmt_cop(ajustado_2026.sum())}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    show_df(presupuesto_2026_df, money_cols=["Sugerido modelo 2026", f"Ajuste IPC {ipc_2026:.1f}%", "IC 95% inf","IC 95% sup"], key="pres_2026")

    # --- TABLA DE PROYECCIÓN POR LÍNEA PARA 2026 ---
    lineas = sorted(df_noYear["LINEA"].unique())
    tabla_lineas_2026 = []
    for i, mes in enumerate(meses_2026):
        fila = {"Mes": mes.strftime("%b-%Y")}
        prop_df = df_noYear[(df_noYear['FECHA'] >= mes - pd.DateOffset(months=11)) & (df_noYear['FECHA'] <= mes)]
        suma_linea = prop_df.groupby("LINEA")["IMP_PRIMA"].sum()
        suma_total = suma_linea.sum()
        prop = suma_linea / suma_total if suma_total > 0 else pd.Series([1/len(lineas)]*len(lineas), index=lineas)
        for linea in lineas:
            fila[linea] = int(base_2026.iloc[i] * prop.get(linea, 1/len(lineas)))
        tabla_lineas_2026.append(fila)
    df_lineas_tabla_2026 = pd.DataFrame(tabla_lineas_2026)
    df_lineas_tabla_2026 = df_lineas_tabla_2026[["Mes"] + lineas]
    st.markdown("##### Proyección mensual 2026 por Línea", unsafe_allow_html=True)
    show_df(df_lineas_tabla_2026, money_cols=lineas, key="tabla_linea_2026")

with tabs[3]:
    st.subheader("Panel Ejecutivo · Sensibilidades & Hallazgos")
    st.info("Aquí va el panel ejecutivo, sensibilidades y hallazgos automáticos. (Puedes agregar tu lógica de BI aquí)")
