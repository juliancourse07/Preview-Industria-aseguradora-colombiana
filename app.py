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

# Tema base + estilos
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
/* Tooltips sin JS */
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

# Sonido futurista al entrar
st.markdown("""
<audio id="intro-audio" src="https://cdn.pixabay.com/download/audio/2024/01/09/audio_ee3a8b2b42.mp3?filename=futuristic-digital-sweep-168473.mp3"></audio>
<script>
  const a = document.getElementById('intro-audio');
  if (a && !window._aseguraview_sound) { window._aseguraview_sound = true;
    setTimeout(()=>{ a.volume = 0.45; a.play().catch(()=>{}); }, 400); }
</script>
""", unsafe_allow_html=True)

# ======= BRANDING =======
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
st.caption("Forecast mensual (SARIMAX), cierre estimado 2025 y presupuesto sugerido 2026 por Año / Sucursal / Línea / Compañía.")

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

# ============ CARGA ============
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
years = sorted(df['ANIO'].dropna().unique())
year_sel = st.sidebar.multiselect("Año:", years, default=years,
                                  help="Filtra años a mostrar en tablas y gráficas históricas.")
suc_opts  = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts= ["TODAS"] + sorted(df['LINEA'].dropna().unique())    if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc  = st.sidebar.selectbox("Código y Sucursal:", suc_opts,
                            help="Selecciona una sucursal en particular o mantén TODAS para el consolidado.")
lin  = st.sidebar.selectbox("Línea:", linea_opts,
                            help="Filtra por línea de negocio. Ej: AUTOMOVILES, SOAT, VIDA, etc.")
comp = st.sidebar.selectbox("Compañía:", comp_opts,
                            help="Elige una compañía o usa TODAS para ver el total.")

periodos_forecast = st.sidebar.number_input(
    "Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1,
    help="Cantidad de pasos hacia adelante en la vista de Primas. Sugerido: 6."
)

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
      • Integra las bases mensuales de producción (primas) y presupuesto (Imp Prima Cuota).<br>
      • Aplica un modelo <b>SARIMAX estacional</b> (12 meses) para proyectar los meses que faltan del año actual y todo 2026.<br>
      • Limpia meses sin cierre (cero al final) para no “copiar” el último valor y evitar sesgos.<br>
      • Muestra rangos de confianza (IC 95%) y permite exportar a Excel.
      <br><br>
      <b>¿Quién puede usarlo?</b><br>
      • Fuerza comercial, product managers, finanzas y dirección. No requiere conocimiento técnico.
      <br><br>
      <b>¿Qué encuentras?</b><br>
      1) <b>Primas (forecast & cierre)</b>: histórico mensual, acumulado, proyección de meses faltantes {i1}<br>
      2) <b>Presupuesto 2026</b>: ejecución vs presupuesto {i2} y sugerencia mensual 2026 {i3}<br>
      3) <b>Modo Director BI</b>: stress-test, escenarios (+/-%) y anomalías detectadas automáticamente {i4}
      <br><br>
      <i>Tip:</i> filtra por <b>Año</b>, <b>Sucursal</b>, <b>Línea</b> y <b>Compañía</b>. Si dejas “TODAS”, verás el consolidado corporativo.
    </div>
    """.format(
        i1=info_badge("Pronóstico de los meses que aún no están cerrados en el año vigente y estimación del cierre anual."),
        i2=info_badge("Compara mes a mes lo presupuestado (Imp Prima Cuota) vs lo ejecutado y proyecta los meses faltantes."),
        i3=info_badge("Propuesta de presupuesto mensual para 2026 calculado con el modelo, con IC 95% y Excel descargable."),
        i4=info_badge("Juega con escenarios ±% sobre el 2026 sugerido, descarga Excel y revisa anomalías para explicar picos/caídas.")
    ), unsafe_allow_html=True)

    st.markdown("### ¿Cómo leer los gráficos?")
    st.markdown("""
    - Los puntos muestran **valores mensuales**; la línea punteada “**2024**” permite comparar contra el año previo.<br>
    - El área sombreada/curvas de **IC 95%** muestran el rango de confianza del modelo.<br>
    - En **acumulado**, la curva crece mes a mes y el punto final refleja el **cierre proyectado**.
    """, unsafe_allow_html=True)
    st.info("En 1 minuto: filtra por tu Sucursal/Línea, mira el forecast de meses faltantes, valida el cierre del año y baja el Excel. ¡Listo!")

# --------- TAB PRIMAS ---------
with tabs[1]:
    ref_year = int(df['FECHA'].max().year)
    last_real_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_faltantes = max(0, 12 - last_real_month)

    serie_train = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    hist_df, fc_df, smape6 = fit_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6)

    # Serie 2024 para comparación
    serie_2024 = ensure_monthly(serie_prima_all[serie_prima_all.index.year == 2024])
    df_2024 = pd.DataFrame({"FECHA": serie_2024.index, "Mensual_2024": serie_2024.values})
    df_2024["ACUM_2024"] = serie_2024.cumsum().values

    # KPIs
    ytd_ref = serie_prima_all[serie_prima_all.index.year == ref_year].sum()
    cierre_ref = ytd_ref + (fc_df['Forecast_mensual'].head(meses_faltantes).sum() if meses_faltantes>0 else 0.0)

    # --- Cierre anual 2024 ---
    cierre_2024 = float(serie_2024.sum()) if not serie_2024.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"YTD {ref_year}", fmt_cop(ytd_ref))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {ref_year}", fmt_cop(cierre_ref))
    c4.metric("Cierre anual 2024", fmt_cop(cierre_2024))
    c4.markdown(
        " " + info_badge("Total de primas ejecutadas en 2024 (enero a diciembre). "
                         "Úsalo como base para comparar el rendimiento de 2025."), unsafe_allow_html=True
    )

    # Comparativo 2024 vs cierre estimado ref_year
    st.markdown(f"#### Comparativo rápido: 2024 vs cierre estimado {ref_year}")
    comp_bar = pd.DataFrame({"Año": ["2024", f"Est. {ref_year}"], "Valor": [cierre_2024, cierre_ref]})
    fig_comp = px.bar(comp_bar, x="Año", y="Valor", text="Valor")
    fig_comp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_comp.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Mensual
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="")
    fig_m = nicer_line(fig_m, "Primas mensuales (histórico) y forecast")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    if not df_2024.empty:
        fig_m.add_scatter(x=df_2024["FECHA"], y=df_2024["Mensual_2024"], name="2024 (mensual)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    st.plotly_chart(fig_m, use_container_width=True)

    # Acumulado
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="")
    fig_a = nicer_line(fig_a, "Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    if not df_2024.empty:
        fig_a.add_scatter(x=df_2024["FECHA"], y=df_2024["ACUM_2024"], name="2024 (acum)",
                          mode="lines+markers", line=dict(width=3, dash="dash"), opacity=0.9)
    # Línea de referencia cierre 2024
    if cierre_2024 > 0:
        fig_a.add_hline(y=cierre_2024, line_dash="dot", line_width=2,
                        annotation_text=f"Cierre 2024: {fmt_cop(cierre_2024)}",
                        annotation_position="top left")
        # marcador opcional del último punto
        ultimo_2024 = df_2024.iloc[-1] if not df_2024.empty else None
        if ultimo_2024 is not None:
            fig_a.add_scatter(x=[ultimo_2024["FECHA"]], y=[ultimo_2024["ACUM_2024"]],
                              mode="markers+text", name="Fin 2024",
                              text=[f"{fmt_cop(cierre_2024)}"], textposition="top center",
                              marker=dict(size=9, symbol="diamond"))
    st.plotly_chart(fig_a, use_container_width=True)

    st.markdown(f"### Próximos meses proyectados (no cerrados en {ref_year})")
    if meses_faltantes > 0:
        meses_mostrar = st.slider(f"Meses a listar (faltantes de {ref_year}):", 1, meses_faltantes, min(6, meses_faltantes),
                                  help="Cuántos meses faltantes quieres ver en la tabla de pronóstico.")
        sel = fc_df.head(meses_mostrar).copy()
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })
        show_df(tabla_faltantes, money_cols=["Proyección","IC 95% inf","IC 95% sup"], key="faltantes_2025")
    else:
        st.info(f"No quedan meses por cerrar en {ref_year} con los datos actuales.")

    st.success(f"Así se proyectan los meses faltantes de {ref_year} y el **cierre estimado** del año es **{fmt_cop(cierre_ref)}**.")

    st.caption("Excel listo para enviar por correo " + info_badge("Incluye hojas separadas para histórico y forecast."), unsafe_allow_html=True)
    hist_tbl = hist_df.copy(); hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl   = fc_df.copy();   fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_bytes = to_excel_bytes({"Historico": hist_tbl, f"Forecast {ref_year} completo": fc_tbl})
    st.download_button("⬇️ Descargar Excel (PRIMAS)", data=xls_bytes,
                       file_name="primas_forecast.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[2]:
    st.subheader("Ejecución vs Presupuesto 2025 y Presupuesto sugerido 2026")

    ref_year = int(df['FECHA'].max().year)
    serie_exec = ensure_monthly(serie_prima_all)
    serie_pres = ensure_monthly(serie_presu_all)
    ejec_ref = serie_exec[serie_exec.index.year == ref_year]
    pres_ref = serie_pres[serie_pres.index.year == ref_year]

    this_month = pd.Timestamp.today().month
    ytd_ejec = ejec_ref.loc[ejec_ref.index <= pd.Timestamp(f"{ref_year}-{this_month:02d}-01")].sum()
    ytd_pres = pres_ref.loc[pres_ref.index <= pd.Timestamp(f"{ref_year}-{this_month:02d}-01")].sum() if not pres_ref.empty else np.nan
    var_pct = ((ytd_ejec - ytd_pres) / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

    last_real_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_falt_ref = max(0, 12 - last_real_month)
    serie_exec_clean = sanitize_trailing_zeros(serie_exec, ref_year)
    _, fc_ref, _ = fit_forecast(serie_exec_clean, steps=max(1, meses_falt_ref))
    proy_ref_mensual = fc_ref.set_index("FECHA")["Forecast_mensual"] if meses_falt_ref > 0 else pd.Series(dtype=float)
    cierre_ejec_ref = ejec_ref.sum() + (proy_ref_mensual.sum() if not proy_ref_mensual.empty else 0.0)

    c1,c2,c3 = st.columns(3)
    c1.metric(f"Presupuesto {ref_year} YTD", fmt_cop(ytd_pres) if not np.isnan(ytd_pres) else "s/datos")
    c2.metric(f"Ejecutado {ref_year} YTD", fmt_cop(ytd_ejec),
              delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric(f"Cierre estimado {ref_year} (ejecución)", fmt_cop(cierre_ejec_ref))

    st.markdown("#### ¿Qué ves aquí? " + info_badge(
        "Compara, mes a mes, lo PRESUPUESTADO (Imp Prima Cuota) contra el valor EJECUTADO. "
        "Si un mes aún no se ha cerrado, el sistema lo estima con SARIMAX."
    ), unsafe_allow_html=True)

    comp_ref = pd.DataFrame(index=pd.date_range(f"{ref_year}-01-01", f"{ref_year}-12-01", freq="MS"))
    comp_ref["Presupuesto"] = pres_ref.reindex(comp_ref.index) if not pres_ref.empty else np.nan
    comp_ref["Ejecutado"]   = ejec_ref.reindex(comp_ref.index)
    if meses_falt_ref > 0 and not proy_ref_mensual.empty:
        idx_common = comp_ref.index.intersection(proy_ref_mensual.index)
        if len(idx_common) > 0:
            comp_ref.loc[idx_common, "Proyección ejecución"] = proy_ref_mensual.loc[idx_common].values

    figp = px.line(
        comp_ref.reset_index(names="FECHA"), x="FECHA",
        y=[c for c in ["Presupuesto","Ejecutado","Proyección ejecución"] if c in comp_ref.columns], title=""
    )
    figp = nicer_line(figp, f"{ref_year}: Presupuesto vs Ejecutado y proyección mensual")
    st.plotly_chart(figp, use_container_width=True)

    st.markdown("#### Presupuesto 2026 sugerido " + info_badge(
        "Es la proyección mensual del modelo para 2026. "
        "Tómalo como punto de partida y luego ajusta metas por canal, ciudad o línea."
    ), unsafe_allow_html=True)

    # Robustez: si la serie filtrada es corta, fallback corporativo
    serie_exec_clean_local = serie_exec_clean.copy()
    if len(serie_exec_clean_local.dropna()) < 18:
        st.info("La muestra filtrada es corta para un pronóstico estable. "
                "Se usará el comportamiento global como referencia para el 2026 sugerido.")
        serie_exec_global = ensure_monthly(df.groupby('FECHA')['IMP_PRIMA'].sum().sort_index())
        serie_exec_clean_local = sanitize_trailing_zeros(serie_exec_global, ref_year)

    pasos_total = max(1, meses_falt_ref) + 12
    _, fc_ext, _ = fit_forecast(serie_exec_clean_local, steps=pasos_total, eval_months=6)
    sug_2026 = fc_ext.tail(12).set_index("FECHA"); sug_2026.index = pd.date_range("2026-01-01","2026-12-01",freq="MS")
    presupuesto_2026_df = pd.DataFrame({
        "FECHA": sug_2026.index,
        "Presupuesto sugerido 2026": sug_2026["Forecast_mensual"].round(0).astype(int),
        "IC 95% inf": sug_2026["IC_lo"].round(0).astype(int),
        "IC 95% sup": sug_2026["IC_hi"].round(0).astype(int)
    })
    total_2026 = presupuesto_2026_df["Presupuesto sugerido 2026"].sum()
    st.success(f"**Presupuesto sugerido 2026 (total): {fmt_cop(total_2026)}**")

    show_df(presupuesto_2026_df, money_cols=["Presupuesto sugerido 2026","IC 95% inf","IC 95% sup"], key="pres_2026")
    st.caption("Excel con comparativo y sugerido " + info_badge("Hoja 1: Pres vs Ejec del año. Hoja 2: 2026 sugerido."), unsafe_allow_html=True)
    comp_ref_tbl = comp_ref.reset_index().rename(columns={"index":"FECHA"}); comp_ref_tbl["FECHA"] = comp_ref_tbl["FECHA"].dt.strftime("%Y-%m")
    p2026_tbl = presupuesto_2026_df.copy(); p2026_tbl["FECHA"] = p2026_tbl["FECHA"].dt.strftime("%Y-%m")
    xls_pres = to_excel_bytes({f"{ref_year} Pres vs Ejec": comp_ref_tbl, "2026 Presupuesto sugerido": p2026_tbl})
    st.download_button("⬇️ Descargar Excel (PRESUPUESTO)", data=xls_pres,
                       file_name="presupuesto_refyear_y_2026.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- TAB MODO DIRECTOR BI ---------
with tabs[3]:
    st.subheader("Panel Ejecutivo · Sensibilidades & Hallazgos")
    colA,colB = st.columns([1,1])

    with colA:
        st.markdown("#### Escenario 2026 ajustado " + info_badge(
            "Mueve el porcentaje para ver cómo cambia el total anual si todos los meses de 2026 suben o bajan."
        ), unsafe_allow_html=True)
        ajuste_pct = st.slider("Ajuste vs. sugerido 2026 (±30%)", -30, 30, 0, 1,
                               help="Aplica un porcentaje global de ajuste al 2026 sugerido para crear un escenario.")
        if 'presupuesto_2026_df' not in locals() or presupuesto_2026_df.empty:
            st.info("Primero calcula el presupuesto sugerido 2026 en la pestaña anterior.")
        else:
            base_26 = presupuesto_2026_df.copy()
            base_26["Escenario ajustado 2026"] = (base_26["Presupuesto sugerido 2026"]*(1+ajuste_pct/100)).round(0).astype(int)
            total_base = int(base_26["Presupuesto sugerido 2026"].sum())
            total_adj  = int(base_26["Escenario ajustado 2026"].sum())
            c1,c2 = st.columns(2)
            c1.metric("Total sugerido 2026", fmt_cop(total_base))
            c2.metric("Total escenario 2026", fmt_cop(total_adj), delta=f"{ajuste_pct:+d}%")
            show_df(base_26[["FECHA","Presupuesto sugerido 2026","Escenario ajustado 2026"]], key="escenario26")
            xls_dir = to_excel_bytes({"2026_sugerido_vs_ajustado": base_26.assign(FECHA=base_26["FECHA"].dt.strftime("%Y-%m"))})
            st.download_button("⬇️ Descargar Excel (Modo Director - 2026)", data=xls_dir,
                               file_name="modo_director_2026.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with colB:
        st.markdown("#### Stress test / Tornado " + info_badge(
            "Compara rápidamente 3 escenarios: Base (modelo), -X% y +X%. Mide cuánto cambiaría el total del año."
        ), unsafe_allow_html=True)
        perc = st.select_slider("Rango de sensibilidad", options=[5,10,15,20,25,30], value=10,
                                help="Evalúa el impacto total si todo 2026 se mueve ±X%")
        if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty:
            base_26 = presupuesto_2026_df.copy()
            up = int((base_26["Presupuesto sugerido 2026"]*(1+perc/100)).sum())
            dn = int((base_26["Presupuesto sugerido 2026"]*(1-perc/100)).sum())
            bench = int(base_26["Presupuesto sugerido 2026"].sum())
            tornado = pd.DataFrame({"Escenario":[f"-{perc}%", "Base", f"+{perc}%"], "Total":[dn, bench, up]})
            fig_t = px.bar(tornado, x="Escenario", y="Total", text="Total")
            fig_t.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_t.update_layout(yaxis_title="COP", xaxis_title=None, margin=dict(l=10,r=10,t=20,b=20))
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Calcula primero el sugerido 2026 para ver la sensibilidad.")

    st.markdown("---")
    st.markdown("#### Hallazgos automáticos (anomalías)")
    st.caption("Detección por **z-score ≥ 2.5** sobre la serie mensual suavizada." +
               info_badge("Marcamos picos/caídas inusuales para que el equipo explique causas: campañas, eventos, ajustes contables, etc."),
               unsafe_allow_html=True)
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
        ytd_ref_sum = int(serie_prima_all[serie_prima_all.index.year == yref].sum())
        falt = max(0, 12 - last_actual_month_from_df(df_noYear, yref))
        cierre_ref2 = int(ytd_ref_sum + (fc_df['Forecast_mensual'].head(falt).sum() if 'fc_df' in locals() and falt>0 else 0))
        total_26 = int(presupuesto_2026_df["Presupuesto sugerido 2026"].sum()) if 'presupuesto_2026_df' in locals() and not presupuesto_2026_df.empty else 0
        st.info(f"**Resumen ejecutivo** — Con los datos al corte, el **cierre {yref}** se estima en **{fmt_cop(cierre_ref2)}**. "
                f"Para **2026**, el **presupuesto sugerido** asciende a **{fmt_cop(total_26)}**. "
                "Use el *stress test* y el *escenario ajustado* para validar metas por cartera / región.")
    except:
        pass
