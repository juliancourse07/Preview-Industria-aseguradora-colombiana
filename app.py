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

def safe_int(series):
    # Convierte NaN, inf y -inf a 0 antes de pasar a int
    return pd.Series(series).replace([np.nan, np.inf, -np.inf], 0).round(0).astype(int)

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

st.markdown("""
<audio id="intro-audio" src="https://cdn.pixabay.com/download/audio/2024/01/09/audio_ee3a8b2b42.mp3?filename=futuristic-digital-sweep-168473.mp3"></audio>
<script>
  const a = document.getElementById('intro-audio');
  if (a && !window._aseguraview_sound) { window._aseguraview_sound = true;
    setTimeout(()=>{ a.volume = 0.45; a.play().catch(()=>{}); }, 400); }
</script>
""", unsafe_allow_html=True)

LOGO_URL = "https://d7nxjt1whovz0.cloudfront.net/marketplace/logos/divisions/seguros-de-vida-del-estado.png"
HERO_URL = "https://images.unsplash.com/photo-1556157382-97eda2d62296?q=80&w=2400&auto=format&fit=crop"

st.markdown(f"""
<div class="glass" style="display:flex;align-items:center;gap:18px;margin-bottom:12px">
  <img src="{LOGO_URL}" alt="Seguros del Estado" style="height:48px;object-fit:contain;border-radius:8px;" onerror="this.style.display='none'">
  <div>
    <div class="neon" style="font-size:20px;font-weight:700;">AseguraView · Colombiana Seguros del Estado S.A.</div>
    <div style="opacity:.75">Inteligencia de negocio en tiempo real · Forecast SARIMAX</div>
  </div>
</div>
<img src="{HERO_URL}" alt="hero" style="width:100%;height:180px;object-fit:cover;border-radius:18px;opacity:.35;margin-bottom:10px" onerror="this.style.display='none'">
""", unsafe_allow_html=True)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (SARIMAX), nowcast, cierre estimado 2025 y presupuesto sugerido 2026 (con IPC), por Sucursal / Línea / Compañía.")

# ============ DATOS ============
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

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

def nicer_line(fig: Figure, title: str):
    fig.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=2))
    fig.update_layout(
        title=title, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=list([
            dict(count=12, label="1y", step="month", stepmode="backward"),
            dict(count=24, label="2y", step="month", stepmode="backward"),
            dict(step="all")
        ])
    ))
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

tabs = st.tabs(["🏠 Presentación", "📈 Primas (forecast & cierre)", "🧭 Presupuesto 2026", "🧠 Modo Director BI"])

# --------- TAB PRESENTACIÓN ---------
with tabs[0]:
    st.markdown("## Bienvenido a *AseguraView*")
    st.markdown("""
    <div class="glass" style="padding:18px; line-height:1.5;">
      <b>¿Qué es?</b><br>
      <b>AseguraView</b> es el tablero corporativo para visualizar <b>Primas</b> y comparar la <b>ejecución vs presupuesto</b>,
      además de pronosticar <b>cierres de año</b> y sugerir el <b>presupuesto 2026</b> con base en el comportamiento mensual histórico.
      <br><br>
      <b>¿Cómo lo hace?</b><br>
      • <b>SARIMAX</b> (estacionalidad 12) para meses faltantes del año actual y todo 2026.<br>
      • Limpieza de ceros finales y <b>exclusión del mes actual si está parcial</b> (nowcast) para evitar sesgos a la baja.<br>
      • <b>Presupuesto 2026</b>: se ajusta automáticamente por el <b>IPC proyectado</b> que defines en el panel lateral (<i>{ipc:.1f}%</i>).<br>
      • IC 95%, tablas exportables a Excel y modo ejecutivo con escenarios.
      <br><br>
      <i>Nota:</i> Si el mes en curso no está completo (p. ej., septiembre hasta el día 23), se estima con el modelo (nowcast) y se suma al YTD.
    </div>
    """.format(ipc=ipc_2026), unsafe_allow_html=True)

# --------- TAB PRIMAS (forecast & cierre) ---------
with tabs[1]:
    ref_year = int(df['FECHA'].max().year)
    base_series = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    today = pd.Timestamp.today()
    df_closed_months = df_noYear[(df_noYear['FECHA'].dt.year == ref_year) & (df_noYear['IMP_PRIMA'] > 0)].copy()
    ultimo_mes_cerrado = int(df_closed_months['FECHA'].dt.month.max())
    meses_faltantes = 12 - ultimo_mes_cerrado
    serie_cerrada = serie_prima_all[serie_prima_all.index.year == ref_year]
    serie_cerrada = serie_cerrada[serie_cerrada.index.month <= ultimo_mes_cerrado]
    prod_2025 = float(serie_cerrada.sum())

    hist_df, fc_df, smape6 = fit_forecast(serie_cerrada, steps=meses_faltantes, eval_months=6)
    meses_faltantes_idx = pd.date_range(
        pd.Timestamp(year=ref_year, month=ultimo_mes_cerrado+1, day=1),
        pd.Timestamp(year=ref_year, month=12, day=1),
        freq="MS"
    )
    fc_df = fc_df.copy()
    fc_df = fc_df[fc_df["FECHA"].isin(meses_faltantes_idx)]

    nowcast_actual = None
    if (today.month == ultimo_mes_cerrado + 1) and (today.day < 28) and len(fc_df) > 0:
        nowcast_actual = float(fc_df.iloc[0]["Forecast_mensual"])

    cierre_est = prod_2025 + fc_df['Forecast_mensual'].sum()

    c1, c2 = st.columns(2)
    c1.metric(
        f"Producción {ref_year} (meses cerrados)",
        fmt_cop(prod_2025),
        help=f"Corresponde a la suma de los meses cerrados: enero a {pd.Timestamp(year=ref_year, month=ultimo_mes_cerrado, day=1).strftime('%B')}"
    )
    c2.metric("Cierre estimado "+str(ref_year), fmt_cop(cierre_est))

    st.markdown("#### Proyección mensual por Línea (meses faltantes en "+str(ref_year)+")")
    prop = df_noYear[
        (df_noYear['FECHA'] >= pd.Timestamp(year=ref_year, month=ultimo_mes_cerrado, day=1) - pd.DateOffset(months=11)) &
        (df_noYear['FECHA'] <= pd.Timestamp(year=ref_year, month=ultimo_mes_cerrado, day=1))
    ].groupby("LINEA")["IMP_PRIMA"].sum()
    if not prop.empty: prop = prop / prop.sum()
    else: prop = pd.Series(1, index=df_noYear['LINEA'].unique()) / len(df_noYear['LINEA'].unique())
    proy_linea = {}
    for _, row in fc_df.iterrows():
        fecha = row["FECHA"].strftime("%b-%Y")
        # safe_int para cada columna por línea
        proy_linea[fecha] = safe_int(row["Forecast_mensual"] * prop)
    cierre_linea_mes_faltantes = pd.DataFrame(proy_linea).T.fillna(0).astype(int)
    st.dataframe(cierre_linea_mes_faltantes, use_container_width=True, hide_index=False)
    st.caption(f"Se muestran los meses faltantes de {ref_year} (incluyendo diciembre) y la proyección por línea.")

    st.markdown(f"### Próximos meses proyectados (no cerrados en {ref_year})")
    meses_mostrar = st.slider(
        f"Meses a listar (faltantes de {ref_year}):",
        1, len(fc_df), len(fc_df)
    )
    sel = fc_df.head(meses_mostrar).copy()
    tabla_faltantes = pd.DataFrame({
        "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
        "Proyección": safe_int(sel["Forecast_mensual"]),
        "IC 95% inf": safe_int(sel["IC_lo"]),
        "IC 95% sup": safe_int(sel["IC_hi"]),
    })
    show_df(tabla_faltantes, money_cols=["Proyección","IC 95% inf","IC 95% sup"], key="faltantes_2025")
    st.caption(f"Siempre se muestra hasta diciembre. Si hay datos reales solo hasta septiembre, octubre-diciembre son estimados.")

    st.markdown("##### Primas mensuales (histórico y forecast)")
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="Primas mensuales (histórico) y forecast")
    fig_m = nicer_line(fig_m, "")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("##### Primas acumuladas y proyección")
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="Primas acumuladas (histórico) y proyección acumulada")
    fig_a = nicer_line(fig_a, "")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    st.plotly_chart(fig_a, use_container_width=True)

    hist_tbl = hist_df.copy(); hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl   = fc_df.copy();   fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_bytes = to_excel_bytes({"Historico": hist_tbl, f"Forecast {ref_year} completo": fc_tbl})
    st.download_button("⬇️ Descargar Excel (PRIMAS)", data=xls_bytes,
                       file_name="primas_forecast.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[2]:
    st.subheader("Ejecución vs Presupuesto año actual y Presupuesto sugerido 2026")
    st.caption(f"Nota: el presupuesto 2026 aplica un ajuste automático de *IPC proyectado {ipc_2026:.1f}%*.")

    ref_year = int(df['FECHA'].max().year)

    serie_exec = ensure_monthly(serie_prima_all)
    serie_pres = ensure_monthly(serie_presu_all)
    pres_ref = serie_pres[serie_pres.index.year == ref_year]

    df_closed_months = df_noYear[(df_noYear['FECHA'].dt.year == ref_year) & (df_noYear['IMP_PRIMA'] > 0)].copy()
    ultimo_mes_cerrado = int(df_closed_months['FECHA'].dt.month.max())

    ejec_cerrada = serie_exec[serie_exec.index.month <= ultimo_mes_cerrado]
    pres_cerrada = pres_ref[pres_ref.index.month <= ultimo_mes_cerrado]
    ytd_ejec = ejec_cerrada.sum()
    ytd_pres = pres_cerrada.sum()

    meses_falt_ref = 12 - ultimo_mes_cerrado
    _, fc_ref, _ = fit_forecast(ejec_cerrada, steps=meses_falt_ref)
    cierre_ejec_ref = ytd_ejec + fc_ref['Forecast_mensual'].sum()
    var_pct = ((ytd_ejec - ytd_pres) / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

    c1,c2,c3 = st.columns(3)
    c1.metric(f"Presupuesto {ref_year} YTD", fmt_cop(ytd_pres) if not np.isnan(ytd_pres) else "s/datos")
    c2.metric(f"Ejecutado {ref_year} YTD", fmt_cop(ytd_ejec),
              delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric(f"Cierre estimado {ref_year} (ejecución)", fmt_cop(cierre_ejec_ref))

    comp_ref = pd.DataFrame(index=pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS"))
    comp_ref["Presupuesto"] = pres_ref.reindex(comp_ref.index) if not pres_ref.empty else np.nan
    comp_ref["Ejecutado"] = serie_exec.reindex(comp_ref.index)
    if meses_falt_ref > 0 and not fc_ref.empty:
        comp_ref.loc[fc_ref["FECHA"], "Proyección ejecución"] = fc_ref.set_index("FECHA")["Forecast_mensual"]

    figp = px.line(
        comp_ref.reset_index(names="FECHA"), x="FECHA",
        y=[c for c in ["Presupuesto","Ejecutado","Proyección ejecución"] if c in comp_ref.columns], title=""
    )
    figp = nicer_line(figp, f"{ref_year}: Presupuesto vs Ejecutado y proyección")
    st.plotly_chart(figp, use_container_width=True)

    serie_exec_clean_local = ejec_cerrada.copy()
    if len(serie_exec_clean_local.dropna()) < 18:
        st.info("Muestra filtrada corta: usamos comportamiento global como referencia para 2026.")
        serie_exec_global = ensure_monthly(df.groupby('FECHA')['IMP_PRIMA'].sum().sort_index())
        serie_exec_clean_local = sanitize_trailing_zeros(serie_exec_global, ref_year)

    pasos_total = meses_falt_ref + 12
    _, fc_ext, _ = fit_forecast(serie_exec_clean_local, steps=pasos_total, eval_months=6)
    sug_2026 = fc_ext.tail(12).set_index("FECHA"); sug_2026.index = pd.date_range("2026-01-01","2026-12-01",freq="MS")

    base_2026 = safe_int(sug_2026["Forecast_mensual"])
    ipc_factor = 1 + (ipc_2026/100.0)
    ajustado_2026 = safe_int(base_2026 * ipc_factor)

    presupuesto_2026_df = pd.DataFrame({
        "FECHA": base_2026.index,
        "Sugerido modelo 2026": base_2026.values,
        f"Ajuste IPC {ipc_2026:.1f}%": ajustado_2026.values,
        "IC 95% inf": safe_int(sug_2026["IC_lo"]).values,
        "IC 95% sup": safe_int(sug_2026["IC_hi"]).values
    })
    total_base = int(base_2026.sum())
    total_ajust = int(ajustado_2026.sum())
    st.success(f"*Presupuesto 2026* — Base modelo: {fmt_cop(total_base)} · Con IPC {ipc_2026:.1f}%: *{fmt_cop(total_ajust)}*")

    show_df(presupuesto_2026_df, money_cols=["Sugerido modelo 2026", f"Ajuste IPC {ipc_2026:.1f}%", "IC 95% inf","IC 95% sup"], key="pres_2026")

    comp_ref_tbl = comp_ref.reset_index().rename(columns={"index":"FECHA"}); comp_ref_tbl["FECHA"] = comp_ref_tbl["FECHA"].dt.strftime("%Y-%m")
    p2026_tbl = presupuesto_2026_df.copy(); p2026_tbl["FECHA"] = p2026_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_pres = to_excel_bytes({f"{ref_year} Pres vs Ejec": comp_ref_tbl, f"2026 Presupuesto (IPC {ipc_2026:.1f}%)": p2026_tbl})
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
            base_col = base_26.columns[2]
            base_26["Escenario ajustado 2026"] = safe_int(base_26[base_col]*(1+ajuste_pct/100))
            total_base = int(base_26[base_col].sum())
            total_adj  = int(base_26["Escenario ajustado 2026"].sum())
            c1,c2 = st.columns(2)
            c1.metric("Total 2026 (con IPC)", fmt_cop(total_base))
            c2.metric("Total escenario 2026", fmt_cop(total_adj), delta=f"{ajuste_pct:+d}%")
            show_df(base_26[["FECHA",base_col,"Escenario ajustado 2026"]], key="escenario26")
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
    st.caption("Detección por *z-score ≥ 2.5* sobre la serie mensual suavizada." +
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
                    "Valor": safe_int(s.loc[outliers.index]).values,
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
