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
st.caption("Dashboard interactivo con forecast ajustado, análisis espacial y comentarios automáticos tipo BI.")

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

def resumen_analitico_proyeccion(hist, proy):
    if proy is None or proy.empty:
        return "No hay suficiente información para análisis de proyección."
    if len(hist) >= 2:
        ult = hist['Suma de VALOR'].iloc[-1]
        ant = hist['Suma de VALOR'].iloc[-2]
        crec = (ult-ant)/abs(ant) if ant != 0 else None
    else:
        crec = None
    tendencia = "alza" if crec and crec>0.05 else ("baja" if crec and crec<-0.05 else "estable")
    if "ACUM_ANUAL" in proy.columns:
        proy_max = proy['ACUM_ANUAL'].max()
    elif "Forecast_acum" in proy.columns:
        proy_max = proy['Forecast_acum'].max()
    else:
        proy_max = 0
    texto = (
        f"La proyección estima un cierre de {proy_max:,.0f}. "
        f"La tendencia reciente es {tendencia}. "
        f"Se recomienda monitorear la evolución mensual para ajustar estrategias."
    )
    if tendencia == "alza":
        texto += " Existe un riesgo de incremento en el valor futuro, revise causas y prepare estrategias."
    if tendencia == "baja":
        texto += " El comportamiento indica una posible reducción, identifique oportunidades de mejora."
    return texto

def resumen_analitico_espacial(df, tipo_analisis):
    if df.empty:
        return "No hay datos para analizar en la selección actual."
    c = df.groupby('CIUDAD')['Suma de VALOR'].sum().sort_values(ascending=False)
    top = c.head(3)
    total = c.sum()
    tipo = "siniestros" if "siniestro" in tipo_analisis.lower() else "primas"
    tendencia = "concentrada" if top.values[0] > 0.3*total else "dispersa"
    mayor = top.idxmax()
    return (
        f"El análisis espacial para {tipo} muestra que la mayor concentración está en {mayor} "
        f"({top.iloc[0]:,.0f}) representando el {top.iloc[0]/total:.1%} del total. "
        f"La distribución es {tendencia}, con las siguientes ciudades destacadas: {', '.join([f'{k} ({v:,.0f})' for k,v in top.items()])}."
    )

df = load_data()

st.sidebar.header("Filtros")
tipo_opts = sorted(df['Primas/Siniestros'].dropna().unique())
tipo = st.sidebar.selectbox("Tipo (Primas/Siniestros):", tipo_opts)
compania_opts = ["Todas"] + sorted(df['COMPAÑÍA'].dropna().unique())
compania = st.sidebar.selectbox("Compañía:", compania_opts)
ciudad_opts = ["Todas"] + sorted(df['CIUDAD'].dropna().unique())
ciudad = st.sidebar.selectbox("Ciudad:", ciudad_opts)
ramo_opts = ["Todas"] + sorted(df['RAMOS'].dropna().unique())
ramo = st.sidebar.selectbox("Ramo:", ramo_opts)

filt = (df['Primas/Siniestros'] == tipo)
if compania != "Todas":
    filt &= (df['COMPAÑÍA'] == compania)
if ciudad != "Todas":
    filt &= (df['CIUDAD'] == ciudad)
if ramo != "Todas":
    filt &= (df['RAMOS'] == ramo)
df_sel = df[filt].sort_values("FECHA").copy()

st.sidebar.markdown("---")
max_periodos = 24
st.sidebar.markdown(f"Máximo periodos proyectables: {max_periodos}")
periodos_forecast = st.sidebar.number_input(
    "¿Cuántos meses adelante quieres proyectar?",
    min_value=1,
    max_value=max_periodos,
    value=6,
    step=1
)

df_sel['AÑO'] = df_sel['FECHA'].dt.year
df_sel['MES'] = df_sel['FECHA'].dt.month
df_sel['ACUM_ANUAL'] = df_sel.groupby(['AÑO'])['Suma de VALOR'].cumsum()

with st.container():
    st.subheader("Proyección personalizada")
    ytd = df_sel[df_sel['FECHA'].dt.year == df_sel['FECHA'].max().year]['Suma de VALOR'].sum()
    proy, cierre_estimado = 0, ytd

    # Limita el número de años para ARIMA (por ejemplo, últimos 3 años)
    if len(df_sel) >= 6 and periodos_forecast > 0:
        df_sel_recent = df_sel.copy()
        if df_sel_recent['AÑO'].nunique() > 3:
            max_year = df_sel_recent['AÑO'].max()
            df_sel_recent = df_sel_recent[df_sel_recent['AÑO'] >= max_year-2]
        ts = df_sel_recent.set_index('FECHA')['Suma de VALOR'].asfreq('MS').fillna(0)
        ts_acum = ts.groupby(ts.index.year).cumsum()
        try:
            with st.spinner("Calculando forecast ARIMA..."):
                modelo = ARIMA(ts, order=(1,1,1))
                ajuste = modelo.fit()
                fc = ajuste.get_forecast(steps=periodos_forecast)
                forecast_index = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(), periods=periodos_forecast, freq='MS')
                forecast_vals = fc.predicted_mean.values
                forecast_ci = fc.conf_int(alpha=0.05).values
                last_acum = ts_acum.iloc[-1]
                forecast_acum = np.cumsum(forecast_vals) + last_acum

                fc_df = pd.DataFrame({
                    'FECHA': forecast_index,
                    'Forecast_mensual': forecast_vals,
                    'Forecast_acum': forecast_acum,
                    'IC_lo': forecast_ci[:,0],
                    'IC_hi': forecast_ci[:,1]
                })
                proy = forecast_acum[-1] - last_acum
                cierre_estimado = last_acum + proy
        except Exception as e:
            st.error(f"Error en ARIMA: {e}")
            fc_df = pd.DataFrame()
    else:
        fc_df = pd.DataFrame()

    with st.container():
        st.markdown("#### Análisis automático de la proyección")
        hist = df_sel[['FECHA','Suma de VALOR','ACUM_ANUAL']].copy()
        analisis = resumen_analitico_proyeccion(hist, fc_df if not fc_df.empty else None)
        st.info(analisis)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"YTD {df_sel['FECHA'].max().year}", f"${ytd:,.0f}".replace(",", "."))
    col2.metric(f"Proyección {periodos_forecast} meses", f"${proy:,.0f}".replace(",", "."))
    col3.metric(f"Total Estimado (+proy)", f"${cierre_estimado:,.0f}".replace(",", "."))

    fig = px.line(df_sel, x='FECHA', y='ACUM_ANUAL', title="Acumulado histórico y proyección", labels={'ACUM_ANUAL': 'Acumulado'})
    if not fc_df.empty:
        fig.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast_acum'], mode='lines+markers', name='Forecast (acum)', line=dict(dash='dot', color='blue'))
        fig.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast_acum'], fill='tozeroy', name='Proy. Sombra', opacity=0.2, line=dict(color="blue"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(df_sel, x='FECHA', y='Suma de VALOR', title="Histórico mensual y Proyección", labels={'Suma de VALOR': 'Mensual'})
    if not fc_df.empty:
        fig2.add_scatter(x=fc_df['FECHA'], y=fc_df['Forecast_mensual'], mode='lines+markers', name='Forecast (mensual)', line=dict(dash='dot', color='orange'))
        fig2.add_scatter(x=fc_df['FECHA'], y=fc_df['IC_lo'], mode='lines', name="IC 95% Lower", line=dict(dash='dot', color='gray'))
        fig2.add_scatter(x=fc_df['FECHA'], y=fc_df['IC_hi'], mode='lines', name="IC 95% Upper", line=dict(dash='dot', color='gray'))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Ver tabla de proyección"):
        hist = df_sel[['FECHA','Suma de VALOR','ACUM_ANUAL']].copy()
        hist['Tipo'] = "Histórico"
        if not fc_df.empty:
            fc_tabla = fc_df.copy()
            fc_tabla = fc_tabla.rename(columns={'Forecast_mensual':'Suma de VALOR', 'Forecast_acum':'ACUM_ANUAL'})
            fc_tabla['Tipo'] = "Proyección"
            fc_tabla = fc_tabla[~fc_tabla['FECHA'].isin(hist['FECHA'])]
            proy_tabla = pd.concat([hist, fc_tabla[['FECHA','Suma de VALOR','ACUM_ANUAL','Tipo']]], axis=0).sort_values('FECHA').reset_index(drop=True)
        else:
            proy_tabla = hist
        st.dataframe(proy_tabla, use_container_width=True)

st.header("Radar y Mapa de Siniestralidad por Ciudad")
if st.button("Ver análisis espacial"):
    df_esp = df[df['Primas/Siniestros'].str.lower().str.contains(tipo.lower())].copy()
    df_esp['lat'] = df_esp['CIUDAD'].str.upper().map(lambda c: city_coords.get(c, (None, None))[0])
    df_esp['lon'] = df_esp['CIUDAD'].str.upper().map(lambda c: city_coords.get(c, (None, None))[1])
    df_esp = df_esp.dropna(subset=['lat','lon'])

    with st.container():
        st.markdown("#### Análisis automático espacial")
        analisis_espacial = resumen_analitico_espacial(df_esp, tipo)
        st.info(analisis_espacial)

    top_cities = df_esp.groupby('CIUDAD')['Suma de VALOR'].sum().sort_values(ascending=False).head(10)
    radar_fig = px.line_polar(r=top_cities.values, theta=top_cities.index, line_close=True, title=f"Radar de {tipo} por Ciudad")
    st.plotly_chart(radar_fig, use_container_width=True)
    fig_map = px.density_mapbox(
        df_esp, lat='lat', lon='lon', z='Suma de VALOR', radius=45,
        center=dict(lat=4.5, lon=-74), zoom=4.5, mapbox_style="carto-darkmatter",
        color_continuous_scale='Turbo', opacity=0.7,
        title=f"Mapa de {tipo} (heatmap Colombia, suavizado)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

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
