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

# ============ CONFIGURACIÓN Y TEMA CLARO/OSC ============
st.set_page_config(
    page_title="AseguraView · Primas & Presupuesto",
    layout="wide",
    page_icon=":bar_chart:"
)

st.markdown("""
<style>
body, .stApp { background: #f5f7fa !important; color: #18212f !important; }
@media (prefers-color-scheme: dark) { body, .stApp { background: #101522 !important; color: #f3f7fa !important; } }
.block-container { padding-top: 0.6rem; }
.stDataFrame th, .stDataFrame td {color: #24314e !important;}
@media (prefers-color-scheme: dark) {.stDataFrame th, .stDataFrame td {color: #e5e9f5 !important;}}
.badge{display:inline-block;position:relative;margin-left:6px}
.badge > .q{cursor:help;color:#e43a7a;font-weight:700}
.badge .tip{visibility:hidden;opacity:0;transition:opacity .15s;position:absolute;left:0;bottom:125%;width:320px;z-index:50;background:rgba(15,23,42,.96);color:#e5e7eb;border:1px solid rgba(148,163,184,.35);padding:10px 12px;border-radius:10px;font-size:12.5px;}
.badge:hover .tip{visibility:visible;opacity:1}
.headerbar {background: #fff; border-bottom: 1px solid #e6e8ee; padding: 4px 0 4px 0; margin-bottom:10px;}
@media (prefers-color-scheme: dark) {.headerbar {background: #181f2f; border-bottom: 1px solid #222e49;}}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://d7nxjt1whovz0.cloudfront.net/marketplace/logos/divisions/seguros-de-vida-del-estado.png"
HERO_URL = "https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=2400&auto=format&fit=crop"
st.markdown(f"""
<div class="headerbar" style="display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:20px;">
    <span style="font-size:17px;font-weight:600;color:#e43a7a;">
      <img src="{LOGO_URL}" height="32" style="vertical-align:middle;margin-right:7px;"> 
      AseguraView · Primas & Presupuesto
    </span>
    <span style="color:#222e49;font-size:14px;opacity:.90">
      Forecast mensual (SARIMAX), nowcast, cierre estimado, presupuesto sugerido 2026.
    </span>
  </div>
  <div style="display:flex;gap:16px;">
    <span style="font-size:15px;color:#e43a7a;">🏠 Presentación</span>
    <span style="font-size:15px;color:#e43a7a;">📈 Primas (forecast & cierre)</span>
    <span style="font-size:15px;color:#e43a7a;">🧭 Presupuesto 2026</span>
    <span style="font-size:15px;color:#e43a7a;">🧠 Modo Director BI</span>
  </div>
</div>
<div style="display:flex;align-items:center;gap:18px;margin-bottom:12px">
  <img src="{LOGO_URL}" alt="Seguros del Estado" style="height:48px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none'">
  <div>
    <div style="font-size:20px;font-weight:700;color:#e43a7a">AseguraView · Colombiana Seguros del Estado S.A.</div>
    <div style="opacity:.75;color:#24314e">Inteligencia de negocio en tiempo real · Forecast SARIMAX</div>
  </div>
</div>
<img src="{HERO_URL}" alt="hero" style="width:100%;height:180px;object-fit:cover;border-radius:18px;opacity:.35;margin-bottom:10px" onerror="this.style.display='none'">
""", unsafe_allow_html=True)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (SARIMAX), nowcast del mes en curso, cierre estimado 2025 y presupuesto sugerido 2026 (con IPC), por Sucursal / Línea / Compañía.")

# ============ DATOS ============
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
    d = df.copy()
    if money_cols is not None:
        for c in money_cols:
            if c in d.columns:
                d[c] = d[c].map(fmt_cop)
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

# ============ TABS ============
tabs = st.tabs(["🏠 Presentación", "📈 Primas (forecast & cierre)", "🧭 Presupuesto 2026", "🧠 Modo Director BI"])

# --------- TAB PRESENTACIÓN ---------
with tabs[0]:
    st.markdown("## Bienvenido a *AseguraView*")
    st.markdown("""
    <div class="glass" style="padding:18px; line-height:1.5;">
      <b>¿Qué es?</b><br>
      AseguraView es el tablero corporativo para visualizar <b>Primas</b> y comparar la <b>ejecución vs presupuesto</b>,
      además de pronosticar <b>cierres de año</b> y sugerir el <b>presupuesto 2026</b> con base en el comportamiento mensual histórico.
      <br><br>
      <b>¿Cómo lo hace?</b><br>
      • <b>SARIMAX</b> (estacionalidad 12) para meses faltantes del año actual y todo 2026.<br>
      • Limpieza de ceros finales y <b>exclusión del mes actual si está parcial</b> (nowcast) para evitar sesgos a la baja.<br>
      • <b>Presupuesto 2026</b>: se ajusta automáticamente por el <b>IPC proyectado</b> que defines en el panel lateral (<i>{ipc:.1f}%</i>).<br>
      • IC 95%, tablas exportables a Excel y modo ejecutivo con escenarios.<br>
      <br>
      <span style="color:#e43a7a;font-weight:600">Corte de información: 07/10/2025.</span>
      <br><br>
      <i>Nota:</i> Si el mes en curso no está completo (p. ej., septiembre hasta el día 23), se estima con el modelo (nowcast) y se suma al YTD.
    </div>
    """.format(ipc=ipc_2026), unsafe_allow_html=True)

# --------- TAB PRIMAS (forecast & cierre) ---------
with tabs[1]:
    ref_year = 2025
    produccion_2025 = df_noYear[
        (df_noYear['FECHA'].dt.year == 2025) & (df_noYear['IMP_PRIMA'] > 0)
    ]
    if not produccion_2025.empty:
        ultimo_mes_con_prima = produccion_2025['FECHA'].max()
        suma_acumulada_2025 = produccion_2025.groupby('FECHA')['IMP_PRIMA'].sum().cumsum().iloc[-1]
    else:
        ultimo_mes_con_prima = None
        suma_acumulada_2025 = 0

    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    serie_train, cur_month_ts, had_partial = split_series_excluding_partial_current(base_series, ref_year)
    if had_partial and cur_month_ts is not None:
        last_closed_month = cur_month_ts.month - 1
    else:
        last_closed_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_faltantes = max(0, 12 - last_closed_month)
    hist_df, fc_df, smape6 = fit_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6)

    cierre_2024 = serie_prima_all[serie_prima_all.index.year == 2024].sum()
    nowcast_actual = None
    if had_partial and not fc_df.empty and cur_month_ts is not None:
        if fc_df.iloc[0]["FECHA"] != cur_month_ts:
            fc_df.iloc[0, fc_df.columns.get_loc("FECHA")] = cur_month_ts
        nowcast_actual = float(fc_df.iloc[0]["Forecast_mensual"])
    ytd_ref = suma_acumulada_2025 + (nowcast_actual if nowcast_actual is not None else 0.0)
    if had_partial and nowcast_actual is not None and len(fc_df) > 1:
        resto = fc_df['Forecast_mensual'].iloc[1:].sum()
    else:
        resto = fc_df['Forecast_mensual'].sum()
    cierre_ref = ytd_ref + resto

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <span style="font-size:16px;font-weight:500;">
        Producción 2025 <span class="badge"><span class="q">?</span><span class="tip">Total acumulado de primas hasta el último mes con cierre en 2025.</span></span>
    </span>
    <br>
    <span style="font-size:2em;font-weight:700;">{fmt_cop(suma_acumulada_2025)}</span>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <span style="font-size:16px;font-weight:500;">
        Cierre estimado 2025 <span class="badge"><span class="q">?</span><span class="tip">Cierre estimado sumando el acumulado YTD más las proyecciones (forecast) para los meses faltantes.</span></span>
    </span>
    <br>
    <span style="font-size:2em;font-weight:700;">{fmt_cop(cierre_ref)}</span>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
    <span style="font-size:16px;font-weight:500;">
        Cierre anual 2024 <span class="badge"><span class="q">?</span><span class="tip">Cierre real del año anterior para referencia y comparación.</span></span>
    </span>
    <br>
    <span style="font-size:2em;font-weight:700;">{fmt_cop(cierre_2024)}</span>
    """, unsafe_allow_html=True)
    c4.markdown(f"""
    <span style="font-size:16px;font-weight:500;">
        SMAPE validación <span class="badge"><span class="q">?</span><span class="tip">¿Qué es SMAPE? Error porcentual medio simétrico en la validación rolling de 6 meses.</span></span>
    </span>
    <br>
    <span style="font-size:2em;font-weight:700;">{smape6:.2f}%</span>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    **Último mes con primas en 2025:** {ultimo_mes_con_prima.strftime('%B %Y') if ultimo_mes_con_prima else 'Sin datos'}  
    **Acumulado hasta ese mes:** {fmt_cop(suma_acumulada_2025)} <span class="badge"><span class="q">?</span><span class="tip">Suma acumulada hasta el último mes con cierre de primas en 2025.</span></span>  
    **Cierre proyectado (con nowcast):** {fmt_cop(cierre_ref)} <span class="badge"><span class="q">?</span><span class="tip">Incluye el nowcast para los meses faltantes.</span></span>
    """, unsafe_allow_html=True)

    # --- TABLA GENERAL DE PROYECCIÓN MENSUAL (con nowcast en mes actual y forecast en los siguientes) ---
    meses_diciembre = pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS")
    serie_real = serie_prima_all.reindex(meses_diciembre, fill_value=None)
    proyeccion_mensual = [int(x) for x in fc_df["Forecast_mensual"].values] if not fc_df.empty else []
    valores = []
    for i, mes in enumerate(meses_diciembre):
        real_value = serie_real.get(mes, None)
        # Si hay valor real y es >0 lo mostramos
        if pd.notnull(real_value) and real_value > 0:
            valores.append(real_value)
        # Si es el mes en curso y no ha finalizado, mostrar el nowcast (forecast del primer mes)
        elif mes.month == pd.Timestamp.today().month and mes.year == pd.Timestamp.today().year and len(proyeccion_mensual) > 0:
            valores.append(proyeccion_mensual[0])
        # Si es mes futuro, mostrar proyección
        elif i < len(proyeccion_mensual):
            valores.append(proyeccion_mensual[i])
        else:
            valores.append(0)
    tabla_general = pd.DataFrame({
        "Mes": meses_diciembre.strftime("%b-%Y"),
        "Valor": valores
    })
    show_df(tabla_general, money_cols=["Valor"], key="tabla_general_mes")

    # --- TABLA DE PROYECCIÓN POR LÍNEA (con nowcast en mes actual y forecast en los siguientes) ---
    lineas = sorted(df_noYear["LINEA"].unique())
    tabla_lineas = []
    for i, mes in enumerate(meses_diciembre):
        fila = {"Mes": mes.strftime("%b-%Y")}
        for linea in lineas:
            v = df_noYear[(df_noYear['FECHA'] == mes) & (df_noYear['LINEA'] == linea)]["IMP_PRIMA"].sum()
            # Valor real si existe
            if v > 0:
                fila[linea] = v
            # Si es el mes actual y no ha finalizado, forecast (nowcast)
            elif mes.month == pd.Timestamp.today().month and mes.year == pd.Timestamp.today().year and len(fc_df) > 0:
                prop = df_noYear[(df_noYear['FECHA'] >= mes - pd.DateOffset(months=11)) & (df_noYear['FECHA'] <= mes)].groupby("LINEA")["IMP_PRIMA"].sum()
                prop = prop / prop.sum() if prop.sum() > 0 else pd.Series([1/len(lineas)]*len(lineas), index=lineas)
                fila[linea] = int(fc_df["Forecast_mensual"].iloc[0] * prop.get(linea, 1/len(lineas)))
            # Mes futuro, proyección
            elif i < len(fc_df):
                prop = df_noYear[(df_noYear['FECHA'] >= mes - pd.DateOffset(months=11)) & (df_noYear['FECHA'] <= mes)].groupby("LINEA")["IMP_PRIMA"].sum()
                prop = prop / prop.sum() if prop.sum() > 0 else pd.Series([1/len(lineas)]*len(lineas), index=lineas)
                fila[linea] = int(fc_df["Forecast_mensual"].iloc[i] * prop.get(linea, 1/len(lineas)))
            else:
                fila[linea] = 0
        tabla_lineas.append(fila)
    df_lineas_tabla = pd.DataFrame(tabla_lineas)
    df_lineas_tabla = df_lineas_tabla[["Mes"] + lineas]
    show_df(df_lineas_tabla, money_cols=lineas, key="tabla_linea_mes")

    # --- GRÁFICO DE PRIMAS MENSUALES (HISTÓRICO Y FORECAST) ---
    st.markdown("##### Primas mensuales (histórico y forecast) <span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Puedes deslizar abajo para ver un rango de fechas específico.</span></span>", unsafe_allow_html=True)
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="", labels={"Mensual": "COP"})
    fig_m.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=2))
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
    st.plotly_chart(fig_m, use_container_width=True)

    # --- GRÁFICO DE PRIMAS ACUMULADAS Y PROYECCIÓN ---
    st.markdown("##### Primas acumuladas y proyección <span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Puedes deslizar para comparar periodos históricos.</span></span>", unsafe_allow_html=True)
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="", labels={"ACUM": "COP"})
    fig_a.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=2))
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    st.plotly_chart(fig_a, use_container_width=True)

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[2]:
    st.subheader("Ejecución vs Presupuesto año actual y Presupuesto sugerido 2026")
    st.caption(f"Nota: el presupuesto 2026 aplica un ajuste automático de *IPC proyectado {ipc_2026:.1f}%*.")

    ref_year = 2025
    pres_2025 = serie_presu_all[serie_presu_all.index.year == 2025].sum()
    pres_2024 = serie_presu_all[serie_presu_all.index.year == 2024].sum()

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

    _, fc_ref, _ = fit_forecast(serie_exec_clean, steps=max(1, meses_falt_ref))

    nowcast_ref = None
    if had_partial_ref and not fc_ref.empty and cur_m_ref is not None:
        if fc_ref.iloc[0]["FECHA"] != cur_m_ref:
            fc_ref.iloc[0, fc_ref.columns.get_loc("FECHA")] = cur_m_ref
        nowcast_ref = float(fc_ref.iloc[0]["Forecast_mensual"])

    ytd_ejec_cerrado = serie_exec_clean[serie_exec_clean.index.year == ref_year].sum()
    ytd_ejec = ytd_ejec_cerrado + (nowcast_ref if nowcast_ref is not None else 0.0)
    cierre_primas_proj = cierre_ref
    pres_anual_2025 = pres_2025
    porc_ejec = (cierre_primas_proj / pres_anual_2025 * 100) if pres_anual_2025 > 0 else 0

    c1,c2,c3 = st.columns(3)
    c1.metric(f"Presupuesto 2025 YTD", fmt_cop(pres_2025) if not np.isnan(pres_2025) else "s/datos")
    c2.metric(f"Ejecutado 2025 YTD (con nowcast)", fmt_cop(ytd_ejec))
    c3.metric(f"Proyección cierre primas vs presupuesto", f"{fmt_cop(cierre_primas_proj)} ({porc_ejec:.1f}%)")

    st.markdown(f"""
    <span style="color:#198754;font-weight:600">Presupuesto 2026 — Base modelo: {fmt_cop(pres_2025)} · Presupuesto 2024: {fmt_cop(pres_2024)} · Con IPC {ipc_2026:.1f}%: <b>{fmt_cop(pres_2025 * (1+ipc_2026/100))}</b></span>
    """, unsafe_allow_html=True)

    pasos_total = max(1, meses_falt_ref) + 12
    _, fc_ext, _ = fit_forecast(serie_exec_clean, steps=pasos_total, eval_months=6)
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
    show_df(presupuesto_2026_df, money_cols=["Sugerido modelo 2026", f"Ajuste IPC {ipc_2026:.1f}%", "IC 95% inf","IC 95% sup"], key="pres_2026")

    # --- TABLA DE PROYECCIÓN POR LÍNEA PARA 2026 ---
    lineas = sorted(df_noYear["LINEA"].unique())
    tabla_lineas_2026 = []
    for i, mes in enumerate(pd.date_range("2026-01-01","2026-12-01",freq="MS")):
        fila = {"Mes": mes.strftime("%b-%Y")}
        prop = df_noYear[(df_noYear['FECHA'] >= mes - pd.DateOffset(months=11)) & (df_noYear['FECHA'] <= mes)] \
            .groupby("LINEA")["IMP_PRIMA"].sum()
        prop = prop / prop.sum() if prop.sum() > 0 else pd.Series([1/len(lineas)]*len(lineas), index=lineas)
        for linea in lineas:
            if i < len(base_2026):
                fila[linea] = int(base_2026.iloc[i] * prop.get(linea, 1/len(lineas)))
            else:
                fila[linea] = 0
        tabla_lineas_2026.append(fila)
    df_lineas_tabla_2026 = pd.DataFrame(tabla_lineas_2026)
    df_lineas_tabla_2026 = df_lineas_tabla_2026[["Mes"] + lineas]
    st.markdown("##### Proyección mensual 2026 por Línea <span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Proyección de cada línea para 2026.</span></span>", unsafe_allow_html=True)
    show_df(df_lineas_tabla_2026, money_cols=lineas, key="tabla_linea_2026")

    xls_pres = to_excel_bytes({"Presupuesto 2026": presupuesto_2026_df, "Proyección líneas 2026": df_lineas_tabla_2026})
    st.download_button("⬇️ Descargar Excel (PRESUPUESTO)", data=xls_pres,
                       file_name="presupuesto_2026_ipc.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB MODO DIRECTOR BI ---------
with tabs[3]:
    st.markdown("#### Panel Ejecutivo · Sensibilidades & Hallazgos")
    colA,colB = st.columns([1,1])
    with colA:
        st.markdown("#### Escenario 2026 ajustado <span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Mueve el porcentaje para ver cómo cambia el total anual si todos los meses de 2026 suben o bajan.</span></span>", unsafe_allow_html=True)
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
            show_df(base_26[["FECHA",base_col,"Escenario ajustado 2026"]], money_cols=[base_col,"Escenario ajustado 2026"], key="escenario26")
            xls_dir = to_excel_bytes({"2026_con_IPC_vs_ajustado": base_26.assign(FECHA=base_26["FECHA"].dt.strftime("%Y-%m"))})
            st.download_button("⬇️ Descargar Excel (Modo Director - 2026)", data=xls_dir,
                               file_name="modo_director_2026.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with colB:
        st.markdown("#### Stress test / Tornado <span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Compara 3 escenarios: Base (con IPC), -X% y +X%. Mide cuánto cambiaría el total del año.</span></span>", unsafe_allow_html=True)
        perc = st.select_slider("Rango de sensibilidad", options=[5,10,15,20,25,30], value=10)
        if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty:
            base_26 = presupuesto_2026_df.copy()
            base_col = base_26.columns[2]  # Ajuste IPC
            up = int((base_26[base_col]*(1+perc/100)).sum())
            dn = int((base_26[base_col]*(1-perc/100)).sum())
            bench = int(base_26[base_col].sum())
            tornado = pd.DataFrame({"Escenario":[f"-{perc}%", "Base", f"+{perc}%"], "Total":[dn, bench, up]})
            tornado["Total"] = tornado["Total"].map(fmt_cop)
            fig_t = px.bar(tornado, x="Escenario", y="Total", text="Total")
            fig_t.update_traces(texttemplate="%{text}", textposition="outside")
            fig_t.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10,r=10,t=20,b=20))
            fig_t.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Calcula primero el 2026 con IPC para ver la sensibilidad.")

    st.markdown("---")
    st.markdown("#### Hallazgos automáticos (anomalías)")
    st.caption("Detección por *z-score ≥ 2.5* sobre la serie mensual suavizada.<span class=\"badge\"><span class=\"q\">?</span><span class=\"tip\">Marcamos picos/caídas inusuales para explicar campañas, eventos o ajustes.</span></span>", unsafe_allow_html=True)
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
        yref = ref_year
        base_series2 = sanitize_trailing_zeros(serie_prima_all.copy(), yref)
        serie_train2, cur_ts2, had_part2 = split_series_excluding_partial_current(base_series2, yref)
        falt2 = max(0, 12 - (cur_ts2.month - 1 if had_part2 and cur_ts2 is not None else last_actual_month_from_df(df_noYear, yref)))
        _, fc_tmp2, _ = fit_forecast(serie_train2, steps=max(1, falt2))
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
        st.info(f"*Resumen ejecutivo* — Con nowcast del mes en curso, el *cierre {yref}* se estima en *{fmt_cop(cierre_ref2)}*. "
                f"Para *2026, el **presupuesto (con IPC {ipc_2026:.1f}%)* asciende a *{fmt_cop(total_26)}*.")
    except:
        pass
