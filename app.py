import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(
    page_title="AseguraView · Ciudades & Ramos",
    layout="wide",
    page_icon=":bar_chart:"
)

# --- Estilos visuales ---
st.markdown(
    """
    <style>
    body {background: radial-gradient(1200px 800px at 10% 10%, #0b1220 0%, #0a0f1a 45%, #060a12 100%) !important;}
    .stApp {background-color: #111827;}
    .css-1d391kg {background-color: rgba(255,255,255,0.06)!important;}
    .stDataFrame th, .stDataFrame td {color: #d8e2ff !important;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AseguraView · Ciudades & Ramos")
st.markdown("Dashboard interactivo para análisis de primas/siniestros por compañía, ciudad y ramo.")

# --- Carga de datos ---
st.sidebar.header("Carga de datos")
excel_file = st.sidebar.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

# Definir los nombres de columnas esperados y su orden
expected_columns = [
    'COMPAÑÍA',
    'CIUDAD',
    'Primas/Siniestros',
    'RAMOS',
    'FECHA',
    'Suma de VALOR'
]

def parse_num_co(x):
    try:
        s = str(x).replace('.', '').replace(',', '.').replace(" ", "")
        return float(s)
    except:
        return np.nan

def load_data(file):
    # Lee el excel y normaliza header
    df = pd.read_excel(file)
    # Verifica si tiene todas las columnas requeridas
    cols = [c.strip() for c in df.columns]
    if set(expected_columns).issubset(set(cols)):
        df.columns = cols
        df = df[expected_columns]
    else:
        st.error(f"El archivo debe tener exactamente estas columnas: {expected_columns}")
        return None

    # Limpia datos
    df['Suma de VALOR'] = df['Suma de VALOR'].apply(parse_num_co)
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    return df

df = None
if excel_file is not None:
    df = load_data(excel_file)
    if df is None:
        st.stop()
else:
    st.warning("Por favor, sube un archivo Excel con los títulos correctos.")

if df is not None:
    # --- Filtros sidebar ---
    st.sidebar.header("Filtros")
    tipo_opts = sorted(df['Primas/Siniestros'].dropna().unique())
    tipo = st.sidebar.selectbox("Tipo (Primas/Siniestros):", tipo_opts)

    compania_opts = sorted(df[df['Primas/Siniestros'] == tipo]['COMPAÑÍA'].dropna().unique())
    compania = st.sidebar.selectbox("Compañía:", compania_opts)

    ciudad_opts = sorted(df[(df['Primas/Siniestros'] == tipo) &
                           (df['COMPAÑÍA'] == compania)]['CIUDAD'].dropna().unique())
    ciudad = st.sidebar.selectbox("Ciudad:", ciudad_opts)

    ramo_opts = sorted(df[(df['Primas/Siniestros'] == tipo) &
                         (df['COMPAÑÍA'] == compania) &
                         (df['CIUDAD'] == ciudad)]['RAMOS'].dropna().unique())
    ramo = st.sidebar.selectbox("Ramo:", ramo_opts)

    # Rango de años
    min_year = df['FECHA'].dt.year.min()
    max_year = df['FECHA'].dt.year.max()
    anio = st.sidebar.selectbox("Año para YTD y Cierre:", list(range(max_year, min_year - 1, -1)), index=0)
    horizonte = st.sidebar.slider("Horizonte forecast (meses):", min_value=3, max_value=24, value=6, step=1)

    # --- Filtrado de datos base ---
    filt = (
        (df['Primas/Siniestros'] == tipo) &
        (df['COMPAÑÍA'] == compania) &
        (df['CIUDAD'] == ciudad) &
        (df['RAMOS'] == ramo)
    )
    df_sel = df[filt].sort_values("FECHA").copy()

    # --- KPIs ---
    with st.container():
        st.subheader("KPIs (YTD y Cierre)")
        df_year = df_sel[df_sel['FECHA'].dt.year == anio]
        ytd = df_year['Suma de VALOR'].sum()

        proy = 0
        cierre_estimado = ytd

        # Solo si hay suficiente histórico y meses futuros a proyectar
        if len(df_sel) >= 6:
            ts = df_sel.set_index('FECHA')['Suma de VALOR'].asfreq('MS').fillna(0)
            modelo = ARIMA(ts, order=(1,1,1))
            ajuste = modelo.fit()
            last_month = df_year['FECHA'].dt.month.max() if not df_year.empty else 0
            meses_restantes = 12 - last_month if last_month and last_month < 12 else 0

            if meses_restantes > 0:
                fc = ajuste.get_forecast(steps=meses_restantes)
                proy = fc.predicted_mean.sum()
                cierre_estimado = ytd + proy

        col1, col2, col3 = st.columns(3)
        col1.metric(f"YTD {anio}", f"${ytd:,.0f}".replace(",", "."))
        col2.metric(f"Proyección Resto {anio}", f"${proy:,.0f}".replace(",", "."))
        col3.metric(f"Cierre Estimado {anio}", f"${cierre_estimado:,.0f}".replace(",", "."))

    # --- Serie histórica y forecast ---
    with st.container():
        st.subheader("Serie histórica y Forecast")
        if len(df_sel) >= 6:
            ts = df_sel.set_index('FECHA')['Suma de VALOR'].asfreq('MS').fillna(0)
            modelo = ARIMA(ts, order=(1,1,1))
            ajuste = modelo.fit()
            forecast_steps = horizonte

            # Forecast solo si horizonte > 0
            if forecast_steps > 0:
                forecast_index = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
                fc = ajuste.get_forecast(steps=forecast_steps)
                fc_df = pd.DataFrame({
                    'FECHA': forecast_index,
                    'Forecast': fc.predicted_mean,
                    'lo95': fc.conf_int(alpha=0.05).iloc[:,0],
                    'hi95': fc.conf_int(alpha=0.05).iloc[:,1],
                })
                fig = px.line(df_sel, x='FECHA', y='Suma de VALOR', title="Histórico y Forecast")
                fig.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast'], mode='lines', name='Forecast', line=dict(dash='dot'))
                fig.add_traces([
                    px.scatter(x=fc_df['FECHA'], y=fc_df['lo95'], mode='lines', name="IC 95% Lower").data[0],
                    px.scatter(x=fc_df['FECHA'], y=fc_df['hi95'], mode='lines', name="IC 95% Upper").data[0],
                ])
                fig.update_traces(line=dict(color="#4F8DFD"), selector=dict(name="Forecast"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay meses futuros para proyectar este año.")
        else:
            st.warning("No hay suficiente histórico para forecast (mín 6 meses).")

    # --- EDA rápido ---
    with st.container():
        st.subheader("EDA rápido: variaciones y MA(3)")
        df_eda = df_sel.copy()
        df_eda['Var (%)'] = 100 * (df_eda['Suma de VALOR'] / df_eda['Suma de VALOR'].shift(1) - 1)
        df_eda['MA3'] = df_eda['Suma de VALOR'].rolling(3).mean()
        fig2 = px.line(df_eda, x='FECHA', y=['Var (%)', 'MA3'], title="Variación mensual (%) y Media móvil 3m")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabla base filtrada")
    st.dataframe(df_sel, use_container_width=True)
