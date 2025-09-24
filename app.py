import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =================== CONFIG ===================
st.set_page_config(
    page_title="AseguraView · Primas & Presupuesto",
    layout="wide",
    page_icon=":bar_chart:"
)

st.markdown(
    """
    <style>
    body {background: radial-gradient(1200px 800px at 10% 10%, #0b1220 0%, #0a0f1a 45%, #060a12 100%) !important;}
    .stApp {background-color: #111827;}
    .stDataFrame th, .stDataFrame td {color: #d8e2ff !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (no acumulado), cierre estimado y presupuesto 2026 por Año / Sucursal / Línea / Compañía.")

# =================== FUENTE ===================
# Si cambias la base, actualiza estos dos:
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# =================== UTILIDADES ===================
def parse_number_co(series: pd.Series) -> pd.Series:
    """Convierte '648.306.977,2' -> 648306977.2 y limpia otros formatos."""
    s = series.astype(str)
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monthly(ts: pd.Series) -> pd.Series:
    """Asegura índice mensual MS y completa con interpolación (no ceros)."""
    ts = ts.asfreq("MS")
    return ts.interpolate("linear").fillna(method="bfill").fillna(method="ffill")

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6):
    """
    Modela con SARIMAX(1,1,1)(1,1,1,12) en log1p.
    Devuelve (hist_df, fc_df, smape_val).
    - hist_df: FECHA, Mensual, ACUM
    - fc_df: FECHA, Forecast_mensual, Forecast_acum, IC_lo, IC_hi
    """
    ts = ensure_monthly(ts_m.copy())
    y = np.log1p(ts)

    # Validación temporal (walk-forward corto)
    smapes = []
    start = max(len(y) - eval_months, 12)
    if len(y) >= start + 1:
        for t in range(start, len(y)):
            y_train = y.iloc[:t]
            y_test = y.iloc[t:t+1]
            try:
                m = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                p = r.get_forecast(steps=1).predicted_mean
                smapes.append(smape(np.expm1(y_test.values), np.expm1(p.values)))
            except Exception:
                r = ARIMA(y_train, order=(1,1,1)).fit()
                p = r.get_forecast(steps=1).predicted_mean
                smapes.append(smape(np.expm1(y_test.values), np.expm1(p.values)))
    smape_last = np.mean(smapes) if smapes else np.nan

    # Entrena completo y proyecta
    try:
        m_full = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                         enforce_stationarity=False, enforce_invertibility=False)
        r_full = m_full.fit(disp=False)
        pred = r_full.get_forecast(steps=steps)
        mean = np.expm1(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05))
    except Exception:
        r_full = ARIMA(y, order=(1,1,1)).fit()
        pred = r_full.get_forecast(steps=steps)
        mean = np.expm1(pred.predicted_mean)
        ci = np.expm1(pred.conf_int(alpha=0.05))

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
    hist_df = pd.DataFrame({"FECHA": ts.index, "Mensual": ts.values, "ACUM": hist_acum.values})
    return hist_df, fc_df, smape_last

# =================== CARGA ===================
@st.cache_data(show_spinner=False)
def load_datos(url_csv: str) -> pd.DataFrame:
    df = pd.read_csv(url_csv)
    df.columns = [c.strip() for c in df.columns]

    # Normaliza nombres de columnas esperadas
    rename_map = {
        'Año': 'ANIO', 'ANO': 'ANIO', 'YEAR': 'ANIO',
        'Mes yyyy': 'MES_TXT', 'MES YYYY': 'MES_TXT', 'Mes': 'MES_TXT', 'MES': 'MES_TXT',
        'Codigo y Sucursal': 'SUCURSAL', 'Código y Sucursal': 'SUCURSAL',
        'Linea': 'LINEA', 'Línea': 'LINEA',
        'Compañía': 'COMPANIA', 'COMPAÑÍA': 'COMPANIA', 'COMPANIA': 'COMPANIA',
        'Imp Prima': 'IMP_PRIMA',
        'Imp Prima Cuota': 'IMP_PRIMA_CUOTA'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # FECHA mensual (ignora el día)
    if 'MES_TXT' in df.columns:
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
    else:
        # fallback: si no hay MES_TXT, usar ANIO-01-01
        df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str) + "-01-01", errors='coerce')
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()

    # Numéricos
    if 'IMP_PRIMA' in df.columns:
        df['IMP_PRIMA'] = parse_number_co(df['IMP_PRIMA'])
    if 'IMP_PRIMA_CUOTA' in df.columns:
        df['IMP_PRIMA_CUOTA'] = parse_number_co(df['IMP_PRIMA_CUOTA'])

    # PRESUPUESTO = Imp Prima Cuota (obligatorio)
    if 'IMP_PRIMA_CUOTA' not in df.columns:
        st.stop()  # no seguimos si no existe presupuesto
    df['PRESUPUESTO'] = df['IMP_PRIMA_CUOTA']

    # Limpieza strings
    for c in ['SUCURSAL', 'LINEA', 'COMPANIA']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()

    # Año (derivado de FECHA si no viene)
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year

    # Mantener columnas clave
    keep = [x for x in ['ANIO','FECHA','SUCURSAL','LINEA','COMPANIA','IMP_PRIMA','PRESUPUESTO'] if x in df.columns]
    df = df[keep].dropna(subset=['FECHA']).copy()
    return df

df = load_datos(gsheet_csv(SHEET_ID, SHEET_NAME_DATOS))

# =================== FILTROS DINÁMICOS ===================
st.sidebar.header("Filtros")

years = sorted(df['ANIO'].dropna().unique())
year_sel = st.sidebar.multiselect("Año:", years, default=years)

suc_opts = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts = ["TODAS"] + sorted(df['LINEA'].dropna().unique()) if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc = st.sidebar.selectbox("Código y Sucursal:", suc_opts)
lin = st.sidebar.selectbox("Línea:", linea_opts)
comp = st.sidebar.selectbox("Compañía:", comp_opts)

periodos_forecast = st.sidebar.number_input("Meses a proyectar:", 1, 24, 6, 1)

# Aplica filtros
df_sel = df[df['ANIO'].isin(year_sel)].copy()
if suc != "TODAS" and 'SUCURSAL' in df_sel.columns:
    df_sel = df_sel[df_sel['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_sel.columns:
    df_sel = df_sel[df_sel['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_sel.columns:
    df_sel = df_sel[df_sel['COMPANIA'] == comp]

# Series mensuales (NO ACUMULADO)
serie_prima = df_sel.groupby('FECHA')['IMP_PRIMA'].sum().sort_index() if 'IMP_PRIMA' in df_sel.columns else pd.Series(dtype=float)
serie_presu = df_sel.groupby('FECHA')['PRESUPUESTO'].sum().sort_index() if 'PRESUPUESTO' in df_sel.columns else pd.Series(dtype=float)

# Si no hay datos, salimos con aviso claro
if serie_prima.empty:
    st.warning("No hay datos de IMP_PRIMA con los filtros seleccionados.")
    st.stop()

# =================== TABS ===================
tabs = st.tabs(["📈 Primas (forecast & cierre)", "🧭 Presupuesto (cumplimiento & 2026)"])

# --------- TAB PRIMAS ---------
with tabs[0]:
    st.subheader("Forecast mensual de Primas y cierre del año en curso")

    hist_df, fc_df, smape6 = fit_forecast(serie_prima, steps=periodos_forecast, eval_months=6)

    # Métricas de cierre del año actual
    anio_actual = pd.Timestamp.today().year
    mes_actual = pd.Timestamp.today().month
    ytd = serie_prima[serie_prima.index.year == anio_actual].sum()

    last_idx = serie_prima.index.max()
    pasos_restantes = 12 - last_idx.month if last_idx.year == anio_actual else max(0, 12 - mes_actual + 1)
    cierre_estimado = ytd + (fc_df['Forecast_mensual'].head(pasos_restantes).sum() if pasos_restantes > 0 else 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"YTD {anio_actual}", f"${ytd:,.0f}".replace(",", "."))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {anio_actual}", f"${cierre_estimado:,.0f}".replace(",", "."))

    # Gráfico mensual (NO ACUMULADO) + forecast
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="Primas mensuales (histórico) y forecast")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"],
                          mode="lines+markers", name="Forecast (mensual)", line=dict(dash="dot"))
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], mode="lines", name="IC 95% inf", opacity=0.4)
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], mode="lines", name="IC 95% sup", opacity=0.4)
    st.plotly_chart(fig_m, use_container_width=True)

    # Gráfico acumulado para referencia visual
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"],
                          mode="lines+markers", name="Forecast (acum)", line=dict(dash="dot"))
    st.plotly_chart(fig_a, use_container_width=True)

# --------- TAB PRESUPUESTO ---------
with tabs[1]:
    st.subheader("Cumplimiento del Presupuesto (año actual) y Presupuesto sugerido 2026")

    # Alinea presupuesto a MS y calcula acumulados
    if not serie_presu.empty:
        serie_presu = ensure_monthly(serie_presu)
    else:
        st.info("No hay columna de presupuesto. Recuerda: **Imp Prima Cuota** es el presupuesto (PRESUPUESTO).")
        serie_presu = pd.Series(dtype=float)

    anio_actual = pd.Timestamp.today().year
    mes_actual = pd.Timestamp.today().month

    ejec_act = ensure_monthly(serie_prima[serie_prima.index.year == anio_actual])
    pres_act = ensure_monthly(serie_presu[serie_presu.index.year == anio_actual]) if not serie_presu.empty else pd.Series(dtype=float)

    ytd_ejec = ejec_act.loc[ejec_act.index <= pd.Timestamp(f"{anio_actual}-{mes_actual:02d}-01")].sum()
    ytd_pres = pres_act.loc[pres_act.index <= pd.Timestamp(f"{anio_actual}-{mes_actual:02d}-01")].sum() if not pres_act.empty else np.nan
    var_abs = (ytd_ejec - ytd_pres) if not np.isnan(ytd_pres) else np.nan
    var_pct = (var_abs / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

    # Proyección de cierre vs presupuesto anual
    _, fc_cierre, _ = fit_forecast(serie_prima, steps=max(0, 12 - serie_prima.index.max().month) + 12)
    restante = 12 - serie_prima.index.max().month if serie_prima.index.max().year == anio_actual else 0
    proy_rest = fc_cierre['Forecast_mensual'].head(restante).sum() if restante > 0 else 0.0
    cierre_est = ejec_act.sum() + proy_rest
    presup_anual = pres_act.sum() if not pres_act.empty else np.nan
    gap_cierre = (cierre_est - presup_anual) if not np.isnan(presup_anual) else np.nan
    gap_pct = (gap_cierre / presup_anual * 100) if presup_anual and not np.isnan(presup_anual) and presup_anual != 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Presupuesto YTD", f"${ytd_pres:,.0f}".replace(",", ".") if not np.isnan(ytd_pres) else "s/datos")
    c2.metric("Ejecutado YTD", f"${ytd_ejec:,.0f}".replace(",", "."), delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric(f"Cierre estimado {anio_actual}", f"${cierre_est:,.0f}".replace(",", "."), delta=(f"{gap_pct:+.1f}% vs. Presup." if not np.isnan(gap_pct) else None))
    c4.metric("Presupuesto Anual", f"${presup_anual:,.0f}".replace(",", ".") if not np.isnan(presup_anual) else "s/datos")

    # Acumulado: Presupuesto vs Ejecutado
    comp_df = pd.DataFrame(index=pd.date_range(f"{anio_actual}-01-01", f"{anio_actual}-12-01", freq="MS"))
    comp_df["Presupuesto Acum"] = pres_act.reindex(comp_df.index).fillna(0).cumsum() if not pres_act.empty else np.nan
    comp_df["Ejecutado Acum"] = ejec_act.reindex(comp_df.index).fillna(0).cumsum()

    if restante > 0:
        cierre_acum = comp_df["Ejecutado Acum"].iloc[-1] + fc_cierre['Forecast_mensual'].head(restante).cumsum()
        comp_df["Cierre Est. Acum"] = list(comp_df["Ejecutado Acum"].values) + list(cierre_acum.values)
        comp_df["Cierre Est. Acum"] = comp_df["Cierre Est. Acum"].fillna(method="ffill")

    y_cols = ["Presupuesto Acum", "Ejecutado Acum"] + (["Cierre Est. Acum"] if "Cierre Est. Acum" in comp_df.columns else [])
    figp = px.line(comp_df.reset_index(names="FECHA"), x="FECHA", y=y_cols, title="Cumplimiento vs Presupuesto (acumulado)")
    st.plotly_chart(figp, use_container_width=True)

    # Presupuesto sugerido 2026 (con IC)
    st.markdown("### Presupuesto sugerido 2026 (mensual)")

    pasos_hasta_2026 = (12 - serie_prima.index.max().month) if serie_prima.index.max().year == anio_actual else 0
    pasos_total = pasos_hasta_2026 + 12
    _, fc_ext, _ = fit_forecast(serie_prima, steps=pasos_total, eval_months=6)
    sug_2026 = fc_ext.tail(12).set_index("FECHA")["Forecast_mensual"]
    sug_2026.index = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    ic_lo = fc_ext.tail(12).set_index("FECHA")["IC_lo"].values
    ic_hi = fc_ext.tail(12).set_index("FECHA")["IC_hi"].values

    total_2026 = sug_2026.sum()
    st.metric("Total sugerido 2026", f"${total_2026:,.0f}".replace(",", "."))

    sug_df = pd.DataFrame({
        "FECHA": sug_2026.index,
        "Sugerido 2026": sug_2026.values,
        "IC 95% inf": ic_lo,
        "IC 95% sup": ic_hi
    })
    figb = px.bar(sug_df, x="FECHA", y="Sugerido 2026", title="Presupuesto sugerido 2026 (mensual)")
    figb.add_scatter(x=sug_df["FECHA"], y=sug_df["IC 95% inf"], mode="lines", name="IC 95% inf", opacity=0.5)
    figb.add_scatter(x=sug_df["FECHA"], y=sug_df["IC 95% sup"], mode="lines", name="IC 95% sup", opacity=0.5)
    st.plotly_chart(figb, use_container_width=True)

    with st.expander("Ver tabla 2026"):
        st.dataframe(sug_df, use_container_width=True)

    st.download_button(
        "⬇️ Descargar presupuesto sugerido 2026 (CSV)",
        data=sug_df.to_csv(index=False).encode("utf-8"),
        file_name="presupuesto_sugerido_2026.csv",
        mime="text/csv"
    )

# ------------------ Audio sutil ------------------
st.markdown(
    """
    <audio id="success-audio" src="https://cdn.pixabay.com/audio/2022/10/16/audio_12e1b2d3c3.mp3"></audio>
    <script>
      const audio = document.getElementById('success-audio');
      if(audio) { setTimeout(()=>audio.play(), 1200); }
    </script>
    """,
    unsafe_allow_html=True
)
