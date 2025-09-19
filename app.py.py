# app.py  — AseguraView · Ciudades & Ramos (Excel)
# -------------------------------------------------
# Requisitos (ponlos en requirements.txt del repo):
# streamlit, pandas, numpy, plotly, statsmodels, openpyxl, pillow

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from PIL import Image
import base64, io

# =========================
# Configuración inicial
# =========================
st.set_page_config(
    page_title="AseguraView · Ciudades & Ramos",
    layout="wide",
    page_icon=":bar_chart:"
)

# =========================
# Fondo dinámico (preview)
# =========================
def css_background_image(data_url: str, blur_px: int = 2, darken: float = 0.35) -> str:
    """
    Crea CSS para poner un background full-page con ::before, blur y oscurecimiento.
    data_url: 'url("...")' o 'url(data:image/...)'
    """
    return f"""
    <style>
      .stApp {{
        position: relative;
        background: transparent;
      }}
      .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: {data_url};
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        filter: blur({blur_px}px) brightness({1 - darken});
        transform: scale(1.03);
        z-index: -1;
      }}
      .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
      .stMetric {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 8px;
      }}
      .stDataFrame th, .stDataFrame td {{ color: #d8e2ff !important; }}
    </style>
    """

def css_background_animated() -> str:
    """Gradiente animado tech (fallback si no hay imagen)."""
    return """
    <style>
      .stApp {
        position: relative;
        background: radial-gradient(1200px 800px at 10% 10%, #0b1220 0%, #0a0f1a 45%, #060a12 100%) !important;
        overflow: hidden;
      }
      .stApp::before {
        content: "";
        position: fixed; inset: 0; z-index: -1;
        background: linear-gradient(120deg, #0b1220, #12305a, #0b1220, #1a2a4a);
        background-size: 300% 300%;
        animation: move 18s ease-in-out infinite;
        opacity: .35;
      }
      @keyframes move {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; }
      .stMetric {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 8px;
      }
      .stDataFrame th, .stDataFrame td { color: #d8e2ff !important; }
    </style>
    """

def file_to_data_url(file) -> str:
    """Convierte archivo binario de imagen a data URL base64 (JPEG)."""
    img = Image.open(file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'url("data:image/jpeg;base64,{b64}")'

# =========================
# Sidebar: Fondo
# =========================
with st.sidebar:
    st.header("Fondo")
    bg_mode = st.radio("Modo:", ["Animado (gradiente)", "Imagen subida", "Imagen por URL"], index=0)
    bg_css = ""
    if bg_mode == "Imagen subida":
        bg_file = st.file_uploader("Sube imagen (JPG/PNG)", type=["jpg","jpeg","png"], key="bgfile")
        if bg_file:
            bg_css = css_background_image(file_to_data_url(bg_file), blur_px=2, darken=0.35)
    elif bg_mode == "Imagen por URL":
        url = st.text_input("URL de imagen (https://...)", value="")
        if url.strip():
            bg_css = css_background_image(f'url("{url.strip()}")', blur_px=2, darken=0.35)

# Inyecta CSS del fondo
if bg_mode == "Animado (gradiente)" or not bg_css:
    st.markdown(css_background_animated(), unsafe_allow_html=True)
else:
    st.markdown(bg_css, unsafe_allow_html=True)

# =========================
# Encabezado
# =========================
st.title("AseguraView · Ciudades & Ramos")
st.caption("Dashboard interactivo para análisis de primas/siniestros por Compañía, Ciudad y Ramo.")

# =========================
# Helpers de datos
# =========================
EXPECTED = ['COMPAÑÍA','CIUDAD','Primas/Siniestros','RAMOS','FECHA','Suma de VALOR']
COL_PATTERNS = {
    'COMPAÑÍA': ['compañía','compania','compa','compañia'],
    'CIUDAD': ['ciudad'],
    'Primas/Siniestros': ['primas/siniestros','tipo','primas_siniestros','primas-siniestros'],
    'RAMOS': ['ramos','ramo'],
    'FECHA': ['fecha','periodo','mes'],
    'Suma de VALOR': ['suma de valor','valor','importe','monto']
}

def parse_num_co(x):
    """Formatea números con miles '.' y decimales ',' comunes en CO."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "")
    # Si ya es convertible directo:
    try:
        return float(s)
    except Exception:
        pass
    # Quita miles y convierte decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea columnas 'parecidas' a las esperadas."""
    cols_lower = [c.strip().lower() for c in df.columns]
    mapping = {}
    for target, pats in COL_PATTERNS.items():
        found = None
        for p in pats:
            for i, c in enumerate(cols_lower):
                if p == c or p in c:
                    found = df.columns[i]; break
            if found: break
        if not found:
            raise ValueError(f"No se encontró columna para '{target}'. Encabezados: {list(df.columns)}")
        mapping[target] = found
    return df.rename(columns=mapping)[EXPECTED]

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Lee Excel, ajusta encabezados, parsea número/fecha y limpia nulos."""
    df = pd.read_excel(file, engine="openpyxl")
    try:
        if set(EXPECTED).issubset(set(df.columns)):
            df = df[EXPECTED]
        else:
            df = match_columns(df)
    except Exception as e:
        st.error(f"Encabezados no compatibles: {e}")
        return None
    df['Suma de VALOR'] = df['Suma de VALOR'].apply(parse_num_co)
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['FECHA','Suma de VALOR'])
    return df

def ensure_monthly_series(df_sel: pd.DataFrame) -> pd.DataFrame:
    """Asegura frecuencia mensual (MS) y rellena meses faltantes con 0."""
    if df_sel.empty:
        return df_sel
    s = df_sel.set_index('FECHA')['Suma de VALOR'].sort_index()
    idx = pd.date_range(s.index.min().replace(day=1), s.index.max().replace(day=1), freq='MS')
    s = s.reindex(idx).fillna(0.0)
    return pd.DataFrame({'FECHA': s.index, 'Suma de VALOR': s.values})

def arima_forecast(series_df: pd.DataFrame, steps: int = 6):
    """ARIMA(1,1,1) baseline con bandas al 95%."""
    if series_df.shape[0] < 6 or steps <= 0:
        return None
    x = series_df.set_index('FECHA')['Suma de VALOR']
    model = ARIMA(x, order=(1,1,1))
    fit = model.fit()
    fc = fit.get_forecast(steps=steps)
    ci = fc.conf_int(alpha=0.05)
    lo = ci.iloc[:,0].values
    hi = ci.iloc[:,1].values
    idx = pd.date_range(x.index.max() + pd.offsets.MonthBegin(), periods=steps, freq='MS')
    return pd.DataFrame({'FECHA': idx, 'Forecast': fc.predicted_mean.values, 'lo95': lo, 'hi95': hi})

def money(x: float) -> str:
    """Formato pesos con puntos de miles."""
    return "$" + f"{x:,.0f}".replace(",", ".")

# =========================
# Carga de datos
# =========================
with st.sidebar:
    st.header("Datos")
    excel_file = st.file_uploader("Excel (.xlsx)", type=["xlsx"])

if not excel_file:
    st.warning("Sube el Excel con columnas: " + ", ".join(EXPECTED))
    st.stop()

df = load_data(excel_file)
if df is None or df.empty:
    st.error("No fue posible cargar datos válidos.")
    st.stop()

# =========================
# Filtros
# =========================
with st.sidebar:
    st.header("Filtros")
    tipo_opts = sorted(df['Primas/Siniestros'].dropna().astype(str).unique())
    tipo = st.selectbox("Tipo (Primas/Siniestros):", tipo_opts)

    compania_opts = sorted(df[df['Primas/Siniestros']==tipo]['COMPAÑÍA'].dropna().astype(str).unique())
    compania = st.selectbox("Compañía:", compania_opts)

    ciudad_opts = sorted(df[(df['Primas/Siniestros']==tipo)&(df['COMPAÑÍA']==compania)]['CIUDAD'].dropna().astype(str).unique())
    ciudad = st.selectbox("Ciudad:", ciudad_opts)

    ramo_opts = sorted(df[(df['Primas/Siniestros']==tipo)&(df['COMPAÑÍA']==compania)&(df['CIUDAD']==ciudad)]['RAMOS'].dropna().astype(str).unique())
    ramo = st.selectbox("Ramo:", ramo_opts)

    min_year = int(df['FECHA'].dt.year.min())
    max_year = int(df['FECHA'].dt.year.max())
    anio = st.selectbox("Año para YTD/Cierre:", list(range(max_year, min_year-1, -1)))
    horizonte = st.slider("Horizonte forecast (meses):", 3, 24, 6, 1)

# =========================
# Filtrado base y series
# =========================
mask = (
    (df['Primas/Siniestros']==tipo) &
    (df['COMPAÑÍA']==compania) &
    (df['CIUDAD']==ciudad) &
    (df['RAMOS']==ramo)
)
df_sel = df.loc[mask, ['FECHA','Suma de VALOR']].sort_values('FECHA').copy()
df_monthly = ensure_monthly_series(df_sel)

# =========================
# KPIs (YTD y Cierre)
# =========================
st.subheader("KPIs (YTD y Cierre)")
df_year = df_monthly[df_monthly['FECHA'].dt.year == anio]
ytd = float(df_year['Suma de VALOR'].sum())
proy = 0.0
cierre_estimado = ytd

if df_monthly.shape[0] >= 6:
    last_month = int(df_year['FECHA'].dt.month.max()) if not df_year.empty else 0
    meses_restantes = 12 - last_month if (last_month and last_month < 12) else 0
    if meses_restantes > 0:
        fc_rest = arima_forecast(df_monthly, steps=meses_restantes)
        if fc_rest is not None:
            proy = float(fc_rest['Forecast'].sum())
            cierre_estimado = ytd + proy

c1, c2, c3 = st.columns(3)
c1.metric(f"YTD {anio}", money(ytd))
c2.metric(f"Proyección restante {anio}", money(proy))
c3.metric(f"Cierre estimado {anio}", money(cierre_estimado))

# =========================
# Serie histórica + Forecast
# =========================
st.subheader("Serie histórica y Forecast")
if df_monthly.shape[0] >= 6:
    fc = arima_forecast(df_monthly, steps=horizonte)
    fig = px.line(df_monthly, x='FECHA', y='Suma de VALOR',
                  title=f"{tipo} · {compania} · {ciudad} · {ramo}")
    if fc is not None:
        fig.add_scatter(x=fc['FECHA'], y=fc['Forecast'], mode='lines', name='Forecast', line=dict(dash='dot'))
        fig.add_scatter(x=fc['FECHA'], y=fc['lo95'], mode='lines', name='IC 95% Low', line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc['FECHA'], y=fc['hi95'], mode='lines', name='IC 95% High', line=dict(width=0), showlegend=False)
        # Área entre bandas (ligero)
        area = px.area(fc, x='FECHA', y=['lo95','hi95']).update_traces(opacity=0.12, showlegend=False).data
        fig.add_traces(area)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Histórico insuficiente (mínimo 6 meses).")

# =========================
# EDA rápido
# =========================
st.subheader("EDA rápido: variación mensual y MA(3)")
df_eda = df_monthly.copy()
df_eda['Var (%)'] = 100 * (df_eda['Suma de VALOR'] / df_eda['Suma de VALOR'].shift(1) - 1)
df_eda['MA3']     = df_eda['Suma de VALOR'].rolling(3).mean()
fig2 = px.line(df_eda, x='FECHA', y=['Var (%)','MA3'], title="Variación mensual (%) y Media Móvil 3M")
st.plotly_chart(fig2, use_container_width=True)

# =========================
# Tabla base filtrada
# =========================
st.subheader("Tabla base filtrada")
st.dataframe(df.loc[mask].sort_values('FECHA'), use_container_width=True)
