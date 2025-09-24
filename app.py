# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
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

# ---- Sonido futurista al entrar ----
st.markdown(
    """
    <audio id="intro-audio" src="https://cdn.pixabay.com/download/audio/2024/01/09/audio_ee3a8b2b42.mp3?filename=futuristic-digital-sweep-168473.mp3"></audio>
    <script>
      const a = document.getElementById('intro-audio');
      if (a && !window._aseguraview_sound) {
        window._aseguraview_sound = true;
        setTimeout(()=>{ a.volume = 0.45; a.play().catch(()=>{}); }, 400);
      }
    </script>
    """,
    unsafe_allow_html=True
)

st.title("AseguraView · Primas & Presupuesto")
st.caption("Forecast mensual (SARIMAX), cierre estimado 2025 y presupuesto sugerido 2026 por Año / Sucursal / Línea / Compañía.")

# =================== FUENTE ===================
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"  # <-- cambia si tu archivo es otro
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
    """
    Reindexa a frecuencia mensual y SOLO interpola huecos internos.
    No extiende en los extremos (no ffill/bfill) para no inventar meses cerrados.
    """
    ts = ts.asfreq("MS")
    ts = ts.interpolate(method="linear", limit_area="inside")
    return ts

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def last_actual_month_from_df(df_like: pd.DataFrame, ref_year: int) -> int:
    """
    Último MES (1..12) del ref_year con IMP_PRIMA > 0.
    Si no hay, devuelve 0.
    """
    d = df_like.copy()
    d = d[d['FECHA'].dt.year == ref_year]
    if 'IMP_PRIMA' not in d.columns:
        return 0
    d = d[d['IMP_PRIMA'].fillna(0) > 0]
    if d.empty:
        return 0
    return int(d['FECHA'].max().month)

def sanitize_trailing_zeros(ts: pd.Series, ref_year: int) -> pd.Series:
    """
    Convierte en NaN los ceros consecutivos del FINAL del ref_year y
    recorta la serie hasta el último valor no-NaN para que el forecast
    arranque en el primer mes faltante real (ej. Oct-2025).
    """
    ts = ensure_monthly(ts.copy())
    year_mask = (ts.index.year == ref_year)
    year_series = ts[year_mask]
    if year_series.empty:
        return ts.dropna()

    mask = (year_series[::-1] == 0)
    run, flag = [], True
    for v in mask:
        if flag and bool(v):
            run.append(True)
        else:
            flag = False
            run.append(False)
    trailing_zeros = pd.Series(run[::-1], index=year_series.index)
    ts.loc[trailing_zeros.index[trailing_zeros]] = np.nan

    if ts.last_valid_index() is not None:
        ts = ts.loc[:ts.last_valid_index()]

    return ts.dropna()

def fit_forecast(ts_m: pd.Series, steps: int, eval_months:int=6):
    """SARIMAX(1,1,1)(1,1,1,12) con log1p; fallback ARIMA. Devuelve (hist_df, fc_df, smape_val)."""
    if steps < 1:
        steps = 1
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
    fc_df["IC_lo"] = fc_df["IC_lo"].clip(lower=0)
    fc_df["Forecast_mensual"] = fc_df["Forecast_mensual"].clip(lower=0)

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
        'Año': 'ANIO','ANO':'ANIO','YEAR':'ANIO',
        'Mes yyyy':'MES_TXT','MES YYYY':'MES_TXT','Mes':'MES_TXT','MES':'MES_TXT',
        'Codigo y Sucursal':'SUCURSAL','Código y Sucursal':'SUCURSAL',
        'Linea':'LINEA','Línea':'LINEA',
        'Compañía':'COMPANIA','COMPAÑÍA':'COMPANIA','COMPANIA':'COMPANIA',
        'Imp Prima':'IMP_PRIMA',
        'Imp Prima Cuota':'IMP_PRIMA_CUOTA'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    if 'MES_TXT' in df.columns:
        df['FECHA'] = pd.to_datetime(df['MES_TXT'], dayfirst=True, errors='coerce')
    else:
        df['FECHA'] = pd.to_datetime(df.get('ANIO', pd.Series()).astype(str)+"-01-01", errors='coerce')
    df['FECHA'] = df['FECHA'].dt.to_period("M").dt.to_timestamp()

    if 'IMP_PRIMA' in df.columns:
        df['IMP_PRIMA'] = parse_number_co(df['IMP_PRIMA'])
    if 'IMP_PRIMA_CUOTA' in df.columns:
        df['IMP_PRIMA_CUOTA'] = parse_number_co(df['IMP_PRIMA_CUOTA'])
    else:
        st.error("Falta la columna 'Imp Prima Cuota' (PRESUPUESTO).")
        st.stop()

    df['PRESUPUESTO'] = df['IMP_PRIMA_CUOTA']

    for c in ['SUCURSAL','LINEA','COMPANIA']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    if 'ANIO' not in df.columns:
        df['ANIO'] = df['FECHA'].dt.year

    keep = [x for x in ['ANIO','FECHA','SUCURSAL','LINEA','COMPANIA','IMP_PRIMA','PRESUPUESTO'] if x in df.columns]
    df = df[keep].dropna(subset=['FECHA']).copy()
    return df

df = load_datos(gsheet_csv(SHEET_ID, SHEET_NAME_DATOS))

# =================== FILTROS ===================
st.sidebar.header("Filtros")
years = sorted(df['ANIO'].dropna().unique())
year_sel = st.sidebar.multiselect("Año:", years, default=years)

suc_opts  = ["TODAS"] + sorted(df['SUCURSAL'].dropna().unique()) if 'SUCURSAL' in df.columns else ["TODAS"]
linea_opts= ["TODAS"] + sorted(df['LINEA'].dropna().unique())    if 'LINEA' in df.columns else ["TODAS"]
comp_opts = ["TODAS"] + sorted(df['COMPANIA'].dropna().unique()) if 'COMPANIA' in df.columns else ["TODAS"]

suc  = st.sidebar.selectbox("Código y Sucursal:", suc_opts)
lin  = st.sidebar.selectbox("Línea:", linea_opts)
comp = st.sidebar.selectbox("Compañía:", comp_opts)

periodos_forecast = st.sidebar.number_input("Meses a proyectar (vista PRIMAS):", 1, 24, 6, 1)

# ===== VISTAS (respetan filtros de todo) =====
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

# ===== Serie para forecast/cierre/2026 (ignora filtro de Año; respeta Suc/Línea/Comp) =====
df_noYear = df.copy()
if suc != "TODAS" and 'SUCURSAL' in df_noYear.columns:
    df_noYear = df_noYear[df_noYear['SUCURSAL'] == suc]
if lin != "TODAS" and 'LINEA' in df_noYear.columns:
    df_noYear = df_noYear[df_noYear['LINEA'] == lin]
if comp != "TODAS" and 'COMPANIA' in df_noYear.columns:
    df_noYear = df_noYear[df_noYear['COMPANIA'] == comp]

serie_prima_all = df_noYear.groupby('FECHA')['IMP_PRIMA'].sum().sort_index()
serie_presu_all = df_noYear.groupby('FECHA')['PRESUPUESTO'].sum().sort_index()

ultimo_anio_datos = int(df['FECHA'].max().year)
if ultimo_anio_datos not in year_sel:
    st.warning(f"Tu filtro no incluye el último año con datos ({ultimo_anio_datos}). "
               f"Para faltantes/cierre se usa internamente {ultimo_anio_datos}.")

# =================== TABS ===================
tabs = st.tabs(["📈 Primas (forecast & cierre)", "🧭 Presupuesto 2026"])

# --------- TAB PRIMAS ---------
with tabs[0]:
    ref_year = int(df['FECHA'].max().year)

    last_real_month = last_actual_month_from_df(df_noYear, ref_year)
    meses_faltantes = max(0, 12 - last_real_month)

    serie_train = sanitize_trailing_zeros(serie_prima_all.copy(), ref_year)
    hist_df, fc_df, smape6 = fit_forecast(serie_train, steps=max(1, meses_faltantes), eval_months=6)

    ytd_ref = serie_prima_all[serie_prima_all.index.year == ref_year].sum()
    cierre_ref = ytd_ref + (fc_df['Forecast_mensual'].head(meses_faltantes).sum() if meses_faltantes > 0 else 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"YTD {ref_year}", f"${ytd_ref:,.0f}".replace(",", "."))
    c2.metric("SMAPE validación", f"{smape6:.2f}%" if not np.isnan(smape6) else "s/datos")
    c3.metric(f"Cierre estimado {ref_year}", f"${cierre_ref:,.0f}".replace(",", "."))

    fig_m = px.line(hist_df, x="FECHA", y="Mensual", title="")
    fig_m = nicer_line(fig_m, "Primas mensuales (histórico) y forecast")
    if not fc_df.empty:
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_mensual"], name="Forecast (mensual)", mode="lines+markers")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_lo"], name="IC 95% inf", mode="lines")
        fig_m.add_scatter(x=fc_df["FECHA"], y=fc_df["IC_hi"], name="IC 95% sup", mode="lines")
    st.plotly_chart(fig_m, use_container_width=True)

    fig_a = px.line(hist_df, x="FECHA", y="ACUM", title="")
    fig_a = nicer_line(fig_a, "Primas acumuladas (histórico) y proyección acumulada")
    if not fc_df.empty:
        fig_a.add_scatter(x=fc_df["FECHA"], y=fc_df["Forecast_acum"], name="Forecast (acum)", mode="lines+markers")
    st.plotly_chart(fig_a, use_container_width=True)

    st.markdown(f"### Próximos meses proyectados (no cerrados en {ref_year})")
    if meses_faltantes > 0:
        meses_mostrar = st.slider(f"Meses a listar (faltantes de {ref_year}):", 1, meses_faltantes, min(6, meses_faltantes))
        sel = fc_df.head(meses_mostrar).copy()
        tabla_faltantes = pd.DataFrame({
            "Mes": sel["FECHA"].dt.strftime("%b-%Y"),
            "Proyección": sel["Forecast_mensual"].round(0).astype(int),
            "IC 95% inf": sel["IC_lo"].round(0).astype(int),
            "IC 95% sup": sel["IC_hi"].round(0).astype(int),
        })
        st.dataframe(tabla_faltantes, use_container_width=True)
    else:
        st.info(f"No quedan meses por cerrar en {ref_year} con los datos actuales.")
        tabla_faltantes = pd.DataFrame(columns=["Mes","Proyección","IC 95% inf","IC 95% sup"])

    st.success(
        f"Así se proyectan los meses faltantes de {ref_year} y el **cierre estimado** del año es "
        f"**${cierre_ref:,.0f}**."
        .replace(",", ".")
    )

    hist_tbl = hist_df.copy(); hist_tbl["FECHA"] = hist_tbl["FECHA"].dt.strftime("%Y-%m")
    fc_tbl   = fc_df.copy();   fc_tbl["FECHA"] = fc_tbl["FECHA"].dt.strftime("%Y-%m")
    falt_tbl = tabla_faltantes.copy()
    xls_bytes = to_excel_bytes({
        "Historico": hist_tbl,
        f"Forecast {ref_year} (faltantes)": falt_tbl,
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

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Presupuesto {ref_year} YTD", f"${ytd_pres:,.0f}".replace(",", ".") if not np.isnan(ytd_pres) else "s/datos")
    c2.metric(f"Ejecutado {ref_year} YTD", f"${ytd_ejec:,.0f}".replace(",", "."), delta=(f"{var_pct:+.1f}%" if not np.isnan(var_pct) else None))
    c3.metric(f"Cierre estimado {ref_year} (ejecución)", f"${cierre_ejec_ref:,.0f}".replace(",", "."))

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

    # ===== Presupuesto sugerido 2026 =====
    pasos_total = max(1, meses_falt_ref) + 12
    _, fc_ext, _ = fit_forecast(serie_exec_clean, steps=pasos_total, eval_months=6)
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

    comp_ref_tbl = comp_ref.reset_index().rename(columns={"index":"FECHA"})
    comp_ref_tbl["FECHA"] = comp_ref_tbl["FECHA"].dt.strftime("%Y-%m")
    p2026_tbl = presupuesto_2026_df.copy()
    p2026_tbl["FECHA"] = p2026_tbl["FECHA"].dt.strftime("%Y-%m")

    xls_pres = to_excel_bytes({
        f"{ref_year} Pres vs Ejec": comp_ref_tbl,
        "2026 Presupuesto sugerido": p2026_tbl
    })
    st.download_button(
        "⬇️ Descargar Excel (PRESUPUESTO)",
        data=xls_pres,
        file_name="presupuesto_refyear_y_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
