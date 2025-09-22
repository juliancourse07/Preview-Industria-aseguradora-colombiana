import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import requests

st.set_page_config(
    page_title="AseguraView · Ciudades & Ramos",
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

st.title("AseguraView · Ciudades & Ramos")
st.caption("Dashboard interactivo con forecast ajustado y análisis espacial de siniestralidad.")

# --- Lee de Google Sheets directo ---
SHEET_ID = "1ThVwW3IbkL7Dw_Vrs9heT1QMiHDZw1Aj-n0XNbDi9i8"
SHEET_NAME = "Hoja1"
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

city_coords = {
    "BOGOTA": (4.7110, -74.0721),
    "CALI": (3.4516, -76.5320),
    "CARTAGENA": (10.3910, -75.4794),
    "MEDELLIN": (6.2442, -75.5812),
    "MANIZALES": (5.0689, -75.5174),
    "BARRANQUILLA": (10.9685, -74.7813),
    "BUCARAMANGA": (7.1193, -73.1227),
    "IBAGUE": (4.4389, -75.2322),
    "PEREIRA": (4.8143, -75.6946),
    "CUCUTA": (7.8939, -72.5078),
    "TUNJA": (5.5353, -73.3678),
    "ARMENIA": (4.5339, -75.6811),
    "VALLEDUPAR": (10.4631, -73.2532),
    "NEIVA": (2.9350, -75.2891),
    "PASTO": (1.2136, -77.2811),
    "VILLAVICENCIO": (4.1420, -73.6266),
    "TULUA": (4.0847, -76.1954),
    "SANTA MARTA": (11.2408, -74.1990),
    "YOPAL": (5.3378, -72.3940),
    "POPAYAN": (2.4448, -76.6147),
    "FLORENCIA": (1.6144, -75.6062),
    "MONTERIA": (8.74798, -75.8814),
    "PALMIRA": (3.53944, -76.3036),
    "SINCELEJO": (9.30472, -75.3978),
    "RESTO PAIS": (4.5709, -74.2973),
    "QUIBDO": (5.6947, -76.6611),
    "GIRARDOT": (4.3032, -74.8034),
    "RIOHACHA": (11.5444, -72.9072),
    "BUENAVENTURA": (3.8896, -77.0712),
    "MOCOA": (1.1474, -76.6477),
    "SAN ANDRES": (12.5847, -81.7006),
    "LETICIA": (-4.2153, -69.9406),
    "SJGUAVIARE": (2.5739, -72.6459),
    "PTO. INIRIDA": (3.8653, -67.9239)
}

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        'COMPAÑÍA': 'COMPAÑÍA',
        'COMPANIA': 'COMPAÑÍA',
        'CIUDAD': 'CIUDAD',
        'Primas/Siniestros': 'Primas/Siniestros',
        'Primas/Sinie': 'Primas/Siniestros',
        'RAMOS': 'RAMOS',
        'FECHA': 'FECHA',
        'Suma de VALOR': 'Suma de VALOR',
        'Suma de VALOR ': 'Suma de VALOR'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    expected = ['COMPAÑÍA','CIUDAD','Primas/Siniestros','RAMOS','FECHA','Suma de VALOR']
    df = df[[c for c in expected if c in df.columns]]
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    df['Suma de VALOR'] = pd.to_numeric(df['Suma de VALOR'], errors='coerce')
    df = df.dropna(subset=['FECHA','Suma de VALOR'])
    return df

df = load_data()

# --- Filtros sidebar (agrega opción "Todas") ---
st.sidebar.header("Filtros")
tipo_opts = sorted(df['Primas/Siniestros'].dropna().unique())
tipo = st.sidebar.selectbox("Tipo (Primas/Siniestros):", tipo_opts)
compania_opts = ["Todas"] + sorted(df['COMPAÑÍA'].dropna().unique())
compania = st.sidebar.selectbox("Compañía:", compania_opts)
ciudad_opts = ["Todas"] + sorted(df['CIUDAD'].dropna().unique())
ciudad = st.sidebar.selectbox("Ciudad:", ciudad_opts)
ramo_opts = ["Todas"] + sorted(df['RAMOS'].dropna().unique())
ramo = st.sidebar.selectbox("Ramo:", ramo_opts)

# --- Filtro flexible ---
filt = (df['Primas/Siniestros'] == tipo)
if compania != "Todas":
    filt &= (df['COMPAÑÍA'] == compania)
if ciudad != "Todas":
    filt &= (df['CIUDAD'] == ciudad)
if ramo != "Todas":
    filt &= (df['RAMOS'] == ramo)
df_sel = df[filt].sort_values("FECHA").copy()

# --- Selección de año y mes objetivo, máximo 12 meses adelante ---
min_date = df_sel['FECHA'].min()
max_date = df_sel['FECHA'].max()
años = list(range(min_date.year, max_date.year+2))
anio = st.sidebar.selectbox("Año objetivo", años, index=len(años)-2)
meses = list(range(1, 13))
mes_obj = st.sidebar.selectbox("Mes objetivo", meses, index=max_date.month-1)

# Calcular horizonte de forecast
periodo_obj = datetime(anio, mes_obj, 1)
ult_fecha = df_sel['FECHA'].max()
horizonte = (periodo_obj.year - ult_fecha.year)*12 + (periodo_obj.month - ult_fecha.month)
if horizonte < 0:
    st.warning("No puedes proyectar a una fecha anterior al último dato.")
    st.stop()
elif horizonte == 0:
    proy_meses = 0
else:
    proy_meses = min(12, horizonte)

# --- Serie acumulada por año ---
df_sel['AÑO'] = df_sel['FECHA'].dt.year
df_sel['MES'] = df_sel['FECHA'].dt.month
df_sel['ACUM_ANUAL'] = df_sel.groupby(['AÑO'])['Suma de VALOR'].cumsum()

# --- Forecast ARIMA (acumulado) ---
with st.container():
    st.subheader("Proyección personalizada")
    ytd = df_sel[df_sel['FECHA'].dt.year == anio]['Suma de VALOR'].sum()
    proy = 0
    cierre_estimado = ytd

    if len(df_sel) >= 6 and proy_meses > 0:
        ts = df_sel.set_index('FECHA')['Suma de VALOR'].asfreq('MS').fillna(0)
        ts_acum = ts.groupby(ts.index.year).cumsum()
        modelo = ARIMA(ts_acum, order=(1,1,1))
        ajuste = modelo.fit()
        fc = ajuste.get_forecast(steps=proy_meses)
        forecast_index = pd.date_range(ts_acum.index.max() + pd.offsets.MonthBegin(), periods=proy_meses, freq='MS')
        proy_vals = fc.predicted_mean.values
        proy = proy_vals[-1] if len(proy_vals) else 0
        cierre_estimado = ytd + proy
        fc_df = pd.DataFrame({'FECHA': forecast_index, 'Forecast': proy_vals})
    else:
        fc_df = pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric(f"YTD {anio}", f"${ytd:,.0f}".replace(",", "."))
    col2.metric(f"Proyección hasta {anio}-{mes_obj:02d}", f"${proy:,.0f}".replace(",", "."))
    col3.metric(f"Total Estimado {anio}-{mes_obj:02d}", f"${cierre_estimado:,.0f}".replace(",", "."))

    # --- Serie histórica y forecast (acumulado) ---
    fig = px.line(df_sel, x='FECHA', y='ACUM_ANUAL', title="Histórico Acumulado y Proyección")
    if not fc_df.empty:
        fig.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(dash='dot'))
        fig.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast'], fill='tozeroy', name='Proy. Sombra', opacity=0.2, line=dict(color="blue"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla de proyección"):
        df_proj = pd.concat([
            df_sel[['FECHA','ACUM_ANUAL']].set_index('FECHA'),
            fc_df.set_index('FECHA').rename(columns={'Forecast':'ACUM_ANUAL'})
        ]).sort_index()
        st.dataframe(df_proj, use_container_width=True)

# --- Radar y mapa de siniestralidad (Heatmap) ---
st.header("Radar y Mapa de Siniestralidad por Ciudad")
if st.button("Ver análisis espacial"):
    df_sini = df[df['Primas/Siniestros'] == "Siniestros"].copy()
    df_sini['lat'] = df_sini['CIUDAD'].str.upper().map(lambda c: city_coords.get(c, (None, None))[0])
    df_sini['lon'] = df_sini['CIUDAD'].str.upper().map(lambda c: city_coords.get(c, (None, None))[1])
    df_sini = df_sini.dropna(subset=['lat','lon'])
    # Radar plot
    top_cities = df_sini.groupby('CIUDAD')['Suma de VALOR'].sum().sort_values(ascending=False).head(10)
    radar_fig = px.line_polar(r=top_cities.values, theta=top_cities.index, line_close=True, title="Radar de Siniestros por Ciudad")
    st.plotly_chart(radar_fig, use_container_width=True)
    # Mapa de calor (heatmap)
    fig_map = px.density_mapbox(
        df_sini, lat='lat', lon='lon', z='Suma de VALOR', radius=30,
        center=dict(lat=4.5, lon=-74), zoom=4.5, mapbox_style="carto-positron",
        title="Mapa de siniestralidad (heatmap Colombia)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --- SONIDO DE ÉXITO ---
st.markdown(
    """
    <audio id="success-audio" src="https://cdn.pixabay.com/audio/2022/10/16/audio_12e1b2d3c3.mp3"></audio>
    <script>
      const audio = document.getElementById('success-audio');
      if(audio) { setTimeout(()=>audio.play(), 1500); }
    </script>
    """,
    unsafe_allow_html=True
)
