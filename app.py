import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
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
st.caption("Forecast mensual, cierre estimado 2025 y presupuesto sugerido 2026 por Año / Sucursal / Línea / Compañía.")

# =================== FUENTE ===================
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"  # actualízalo si cambia
SHEET_NAME_DATOS = "Hoja1"

def gsheet_csv(sheet_id, sheet_name):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# =================== UTILIDADES ===================
def parse_number_co(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monthly(ts: pd.Series) -> pd.Series:
    ts = ts.asfreq("MS")
    return ts.interpolate("linear").fillna(method="bfill").fillna(method="ffill")

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6):
    ts = ensure_monthly(ts_m.copy())
    y = np.log1p(ts)

    smapes = []
    start = max(len(y) - eval_months, 12)
    if len(y) >= start + 1:
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

def nicer_line(fig: Figure, title: str):
    fig.update_traces(mode="lines+markers", marker=dict(size=7), line=dict(width=2))
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    for tr in fig.data:
        tr.hovertemplate = "%{x|%b-%Y}<br>%{y:,.0f}<extra></extra>"
    return fig

def to_excel_bytes(sheets: dict) -> bytes:
    """sheets: {'SheetName': DataFrame, ...} -> bytes xlsx"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

# =================== CARGA ===================
@st.cache_data(show_spinner=False)
def load_datos(url_csv: str) -> pd.DataFrame:
    df = pd.read_csv(url_csv)
    df.columns = [c.strip() for c in df.columns]
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

    # FECHA mensual
    if 'MES_TXT' in df.columns:
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
    else:
        df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str) + "-01-01", errors='coerce')
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()

    # Números
    if 'IMP_PRIMA' in df.columns:
        df['IMP_PRIMA'] = parse_number_co(df['IMP_PRIMA'])
    if 'IMP_PRIMA_CUOTA' in df.columns:
        df['IMP_PRIMA_CUOTA'] = parse_number_co(df['IMP_PRIMA_CUOTA'])
    else:
        st.stop()  # presupuesto es obligatorio

    # PRESUPUESTO desde Imp Prima Cuota
    df['PRESUPUESTO'] = df['IMP_PRIMA_CUOTA']

    # Strings y año
    for c in ['SUCURSAL', 'LINEA', 'COMPANIA']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year

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

periodos_forecast = st.sidebar.number_input("Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1)

# Aplica filtros
df_sel = df[df['ANIO'].isin(year_sel)].copy()
if suc != "TODAS" and 'SUCURSAL' in df_sel.columns:
    df_sel = df_sel[df_sel['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_sel.columns:
    df_sel = df_sel[df_sel['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_sel.columns:
    df_sel = df_sel[df_sel['COMPANIA'] == comp]

serie_prima = df_sel.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu = df_sel.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()

if serie_prima.empty:
    st.warning("No hay datos de IMP_PRIMA con los filtros seleccionados.")
    st.stop()

# =================== TABS ===================
tabs = st.tabs(["📈 Primas (forecast & cierre)", "🧭 Presupuesto 2026"])

# --------- TAB PRIMAS ---------
with tabs[0]:
    st.subheader("Forecast de Primas (mensual) y cierre 2025")

    hist_df, fc_df, smape6 = fit_forecast(serie_prima, steps=max(periodos_forecast, 12), eval_months=6)

    anio_actual = pd.Timestamp.today().year
    last_idx = serie_prima.index.max()
    # meses faltantes 2025 (si el último dato ya es 2025)
    faltantes_2025 = 0
    if last_idx.year == anio_actual:
        faltantes_2025 = 12 - last_idx.month
    # cierre estimado 2025
    ytd_2025 = serie_prima[serie_prima.index.year == anio_actual].sum()
    cierre_2025 = ytd_2025 + (fc_df['Forecast_mensual'].head(faltantes_2025).sum() if faltantes_2025 > 0 else 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"YTD {anio_actual}", f"${ytd_2025:,.0f}".replace(",", "."))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {anio_actual}", f"${cierre_2025:,.0f}".replace(",", "."))

    # Gráfico mensual + forecast
    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="")
    fig_m = nicer_line(fig_m, "Primas mensuales (histórico) y forecast")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    st.plotly_chart(fig_m, use_container_width=True)

    # Gráfico acumulado
    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="")
    fig_a = nicer_line(fig_a, "Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    st.plotly_chart(fig_a, use_container_width=True)

    # ---- Tabla "6 próximos meses no cerrados" (dinámica) ----
    st.markdown("### Próximos meses proyectados (no cerrados en 2025)")
    meses_mostrar = st.slider("Meses a listar (faltantes de 2025):", 1, max(faltantes_2025 if faltantes_2025>0 else 6, 6), min(6, max(faltantes_2025, 1)))
    tabla_faltantes = pd.DataFrame(columns=["Mes", "Proyección", "IC 95% inf", "IC 95% sup"])

    if faltantes_2025 > 0:
        sel = fc_df.head(meses_mostrar).copy()
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })
    else:
        st.info("No hay meses faltantes en el año actual con los datos cargados; se muestran los próximos 6 meses de forecast.")
        sel = fc_df.head(6).copy()
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })

    st.dataframe(tabla_faltantes, use_container_width=True)

    # ---- Resumen amigable ----
    st.success(
        f"Según el modelo, **así cierran los meses faltantes de {anio_actual}** y el **cierre estimado** del año es "
        f"**${cierre_2025:,.0f}**. Ajusta estrategias mes a mes según la proyección mostrada."
        .replace(",", ".")
    )

    # ---- Exportar a Excel (PRIMAS) ----
    hist_tbl = hist_df.copy()
    hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl = fc_df.copy()
    fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    falt_tbl = tabla_faltantes.copy()

    xls_bytes = to_excel_bytes({
        "Historico": hist_tbl,
        "Forecast 2025 (faltantes)": falt_tbl,
        "Forecast completo": fc_tbl
    })
    st.download_button(
        "⬇️ Descargar Excel (PRIMAS)",
        data=xls_bytes,
        file_name="primas_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --------- TAB PRESUPUESTO 2026 ---------
with tabs[1]:
    st.subheader("Ejecución vs Presupuesto 2025 y Presupuesto sugerido 2026")

    anio_actual = pd.Timestamp.today().year

    # Alinea series
    serie_prima = ensure_monthly(serie_prima)
    serie_presu = ensure_monthly(serie_presu)

    ejec_2025 = serie_prima[serie_prima.index.year == anio_actual]
    pres_2025 = serie_presu[serie_presu.index.year == anio_actual]

    ytd_ejec = ejec_2025.loc[ejec_2025.index <= pd.Timestamp(f"{anio_actual}-{pd.Timestamp.today().month:02d}-01")].sum()
    ytd_pres = pres_2025.loc[pres_2025.index <= pd.Timestamp(f"{anio_actual}-{pd.Timestamp.today().month:02d}-01")].sum() if not pres_2025.empty else np.nan
    var_pct = ((ytd_ejec - ytd_pres) / ytd_pres * 100) if ytd_pres and not np.isnan(ytd_pres) and ytd_pres != 0 else np.nan

    # Proyección de cierre 2025 del presupuesto (cómo se espera ejecutar mes a mes en 2025)
    _, fc_2025, _ = fit_forecast(serie_prima, steps=max(0, 12 - serie_prima.index.max().month))
    proy_2025_mensual = pd.Series(dtype=float)
    if not fc_2025.empty:
        proy_2025_mensual = fc_2025.set_index("FECHA")["Forecast_mensual"]
    cierre_ejec_2025 = ejec_2025.sum() + proy_2025_mensual.sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Presupuesto 2025 YTD", f"${ytd_pres:,.0f}".replace(",", ".") if not np.isnan(ytd_pres) else "s/datos")
    c2.metric("Ejecutado 2025 YTD", f"${ytd_ejec:,.0f}".replace(",", "."), delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric("Cierre estimado 2025 (ejecución)", f"${cierre_ejec_2025:,.0f}".replace(",", "."))

    # Línea: ejecutado vs presupuesto 2025 (mensual) + proyección de ejecución 2025
    comp_2025 = pd.DataFrame(index=pd.date_range(f"{anio_actual}-01-01", f"{anio_actual}-12-01", freq="MS"))
    comp_2025["Presupuesto 2025"] = pres_2025.reindex(comp_2025.index)
    comp_2025["Ejecutado 2025"] = ejec_2025.reindex(comp_2025.index)
    if not proy_2025_mensual.empty:
        comp_2025.loc[proy_2025_mensual.index, "Proyección ejecución 2025"] = proy_2025_mensual.values

    figp = px.line(comp_2025.reset_index(names="FECHA"), x="FECHA",
                   y=[c for c in ["Presupuesto 2025","Ejecutado 2025","Proyección ejecución 2025"] if c in comp_2025.columns],
                   title="")
    figp = nicer_line(figp, "2025: Presupuesto vs Ejecutado y proyección mensual")
    st.plotly_chart(figp, use_container_width=True)

    # -------- Presupuesto sugerido 2026 (mensual + IC) --------
    pasos_hasta_2026 = (12 - serie_prima.index.max().month) if serie_prima.index.max().year == anio_actual else 0
    pasos_total = pasos_hasta_2026 + 12
    _, fc_ext, _ = fit_forecast(serie_prima, steps=pasos_total, eval_months=6)
    sug_2026 = fc_ext.tail(12).set_index("FECHA")
    sug_2026.index = pd.date_range("2026-01-01", "2026-12-01", freq="MS")

    presupuesto_2026_df = pd.DataFrame({
        "FECHA": sug_2026.index,
        "Presupuesto sugerido 2026": sug_2026["Forecast_mensual"].round(0).astype(int),
        "IC 95% inf": sug_2026["IC_lo"].round(0).astype(int),
        "IC 95% sup": sug_2026["IC_hi"].round(0).astype(int)
    })
    total_2026 = presupuesto_2026_df["Presupuesto sugerido 2026"].sum()
    st.success(f"**Presupuesto sugerido 2026 (total): ${total_2026:,.0f}**".replace(",", "."))

    st.dataframe(presupuesto_2026_df, use_container_width=True)

    figb = px.bar(presupuesto_2026_df, x="FECHA", y="Presupuesto sugerido 2026", title="")
    figb.update_traces(marker_line_width=0.5)
    figb.add_scatter(x=presupuesto_2026_df["FECHA"], y=presupuesto_2026_df["IC 95% inf"], name="IC 95% inf", mode="lines+markers")
    figb.add_scatter(x=presupuesto_2026_df["FECHA"], y=presupuesto_2026_df["IC 95% sup"], name="IC 95% sup", mode="lines+markers")
    for tr in figb.data:
        tr.hovertemplate = "%{x|%b-%Y}<br>%{y:,.0f}<extra></extra>"
    st.plotly_chart(figb, use_container_width=True)

    # ---- Excel exportable (PRESUPUESTO) ----
    # Incluye: 2025 comparativo y 2026 sugerido
    comp_2025_tbl = comp_2025.reset_index().rename(columns={"index":"FECHA"})
    comp_2025_tbl["FECHA"] = comp_2025_tbl["FECHA"].dt.strftime("%Y-%m")
    p2026_tbl = presupuesto_2026_df.copy()
    p2026_tbl["FECHA"] = p2026_tbl["FECHA"].dt.strftime("%Y-%m")

    xls_pres = to_excel_bytes({
        "2025 Pres vs Ejec": comp_2025_tbl,
        "2026 Presupuesto sugerido": p2026_tbl
    })
    st.download_button(
        "⬇️ Descargar Excel (PRESUPUESTO 2025-2026)",
        data=xls_pres,
        file_name="presupuesto_2025_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
