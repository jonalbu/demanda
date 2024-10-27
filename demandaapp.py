import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np


# Configuración de la página de Streamlit
st.set_page_config(layout='wide', initial_sidebar_state='expanded')


st.sidebar.image("Logo/logo.png", use_column_width=True)

with open('style2.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    

st.sidebar.header('Demanda Energética')

# Cargar el archivo de datos
df_demanda = pd.read_csv('Data/demanda_exportada.txt', sep='\t')

# Asegúrate de que la columna 'Date' esté en formato datetime
df_demanda['Date'] = pd.to_datetime(df_demanda['Date'])

# Suma de la demanda por horas (diaria)
df_demanda['demanda_diaria'] = df_demanda.sum(axis=1, skipna=True, numeric_only=True)

# Obtener los 'Values_code' únicos para el filtro
values_code_unicos = df_demanda['Values_code'].unique()

# Rango de fechas mínimo y máximo
min_date = df_demanda['Date'].min()
max_date = df_demanda['Date'].max()

# Widget de rango de fechas
selected_range = st.sidebar.date_input(
    "Selecciona el rango de fechas entre 2021 y 2023",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Widget para seleccionar 'Regiones'
selected_values_code = st.sidebar.multiselect(
    "Selecciona Regiones",
    options=values_code_unicos,
    default=values_code_unicos[:1]  # Selecciona la primera región por defecto
)

# Filtrar datos según el rango de fechas y 'Values_code' seleccionados
df_filtrado = df_demanda[
    (df_demanda['Date'] >= pd.to_datetime(selected_range[0])) & 
    (df_demanda['Date'] <= pd.to_datetime(selected_range[1])) & 
    (df_demanda['Values_code'].isin(selected_values_code))
]

# --- Gráfico de demanda mensual ---
df_filtrado_mensual = df_filtrado.groupby([pd.Grouper(key='Date', freq='M'), 'Values_code'])['demanda_diaria'].sum().reset_index()

fig_mensual = go.Figure()
for code in selected_values_code:
    df_code = df_filtrado_mensual[df_filtrado_mensual['Values_code'] == code]
    fig_mensual.add_trace(go.Scatter(
        x=df_code['Date'].dt.strftime('%Y-%m'),
        y=df_code['demanda_diaria'],
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=7),
        name=f'Demanda Región {code}'
    ))

fig_mensual.update_layout(
    title='Demanda Mensual por Región',
    xaxis=dict(title='Mes', tickangle=-45),
    yaxis=dict(title='Demanda en GWh'),
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    hovermode='x unified'
)

st.title("Visualización de la Demanda Mensual por Región")
st.plotly_chart(fig_mensual, use_container_width=True)

# --- Gráfico de demanda diaria ---
df_filtrado_diario = df_filtrado.groupby([pd.Grouper(key='Date', freq='D'), 'Values_code'])['demanda_diaria'].sum().reset_index()

fig_diario = go.Figure()
for code in selected_values_code:
    df_code_diario = df_filtrado_diario[df_filtrado_diario['Values_code'] == code]
    fig_diario.add_trace(go.Scatter(
        x=df_code_diario['Date'],
        y=df_code_diario['demanda_diaria'],
        mode='lines',
        name=f'Región {code}',
        line=dict(width=2)
    ))

fig_diario.update_layout(
    title='Demanda Energética Diaria por Región',
    xaxis=dict(title='Tiempo (Días)', tickangle=-45),
    yaxis=dict(title='Demanda en GWh'),
    template='plotly_white',
    font=dict(size=14),
    hovermode='x unified'
)

st.title("Visualización de la Demanda Energética Diaria por Región")
st.plotly_chart(fig_diario, use_container_width=True)

# --- Gráfico de demanda diaria superpuesta (Año Ficticio) ---
df_filtrado['day_of_year'] = df_filtrado['Date'].dt.dayofyear
df_filtrado['year'] = df_filtrado['Date'].dt.year

años_disponibles = df_filtrado['year'].unique()
años_seleccionados = st.sidebar.multiselect(
    "Selecciona los Años",
    options=sorted(años_disponibles),
    default=sorted(años_disponibles)[:1]  # Selecciona el primer año por defecto
)

def filtrar_por_años(años):
    df_filtrado_año = df_filtrado[df_filtrado['year'].isin(años)]
    df_combinado = df_filtrado_año.groupby(['day_of_year', 'year'])['demanda_diaria'].sum().reset_index()
    return df_combinado

if años_seleccionados:
    df_filtrado_años = filtrar_por_años(años_seleccionados)

    fig_superpuesto = go.Figure()
    for año in años_seleccionados:
        df_año = df_filtrado_años[df_filtrado_años['year'] == año]
        fig_superpuesto.add_trace(go.Scatter(
            x=df_año['day_of_year'],
            y=df_año['demanda_diaria'],
            mode='lines',
            name=f'Demanda Diaria {año}'
        ))

    fig_superpuesto.update_layout(
        title=f'Demanda Energética Diaria Superpuesta (Año Ficticio) - {", ".join(map(str, años_seleccionados))}',
        xaxis=dict(
            title='Día del Año (1-365)',
            tickmode='linear',
            dtick=30,  # Mostrar una etiqueta cada 30 días
            tickvals=list(range(1, 366, 30)),
            tickangle=0
        ),
        yaxis=dict(title='Demanda en GWh'),
        template='plotly_dark',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.title("Visualización de la Demanda Energética Diaria Superpuesta")
    st.plotly_chart(fig_superpuesto, use_container_width=True)
else:
    st.write("Por favor, selecciona al menos un año para visualizar la demanda diaria.")
    
    
#DATOS DE OFERTA    

# Cargar el archivo de datos
df_oferta = pd.read_csv('Data/oferta_recursos.txt', sep='\t')

# Asegúrate de que la columna 'Date' esté en formato datetime
df_oferta['Date'] = pd.to_datetime(df_oferta['Date'])

# Reemplazar valores nulos por cero
df_oferta = df_oferta.fillna(0)

# Suma de la oferta por horas (diaria)
df_oferta['oferta_diaria'] = df_oferta.sum(axis=1, skipna=True, numeric_only=True)

# Definir los recursos específicos de la región Caribe
recursos_caribe = ["2VJS", "3ENA", "3ENE", "3GPZ", "3HBN", "3HF5", "3HWM", "3IZ6", 
                   "3J2B", "3IS2", "3J2H", "3J4D", "3K6T", "3KJK", "3NLZ", "CTG1", 
                   "CTG2", "CTG3", "EPFV", "GE32", "GEC3", "MATA", "PRG1", "PRG2", 
                   "TBQ3", "TBQ4", "TBST", "TCBE", "TCDT", "TFL1", "TFL4", "TGJ1", 
                   "TGJ2", "TMB1", "TRN1", "URA1"]

# Filtrar el DataFrame por los recursos de la región Caribe
df_region_caribe = df_oferta[df_oferta['Values_code'].isin(recursos_caribe)]

# --- Sidebar para selección de rango de fechas y años ---
st.sidebar.header("Oferta Energética")

# Rango de fechas mínimo y máximo
min_date = df_region_caribe['Date'].min()
max_date = df_region_caribe['Date'].max()

# Widget para seleccionar el rango de fechas
selected_range = st.sidebar.date_input(
    "Selecciona el rango de fechas Entre 2021 y 2023",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filtrar el DataFrame según el rango de fechas seleccionado
df_filtrado = df_region_caribe[
    (df_region_caribe['Date'] >= pd.to_datetime(selected_range[0])) & 
    (df_region_caribe['Date'] <= pd.to_datetime(selected_range[1]))
]

# Obtener los años disponibles
años_disponibles = df_region_caribe['Date'].dt.year.unique()

# Widget para seleccionar los años para el gráfico ficticio
selected_years = st.sidebar.multiselect(
    "Selecciona los años para el segundo gráfico",
    options=sorted(años_disponibles),
    default=[2021]  # Selecciona 2021 por defecto
)

# --- Gráfico #1: Oferta energética diaria en el rango de fechas seleccionado ---
df_filtrado_diario = df_filtrado.groupby('Date')['oferta_diaria'].sum().reset_index()

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df_filtrado_diario['Date'],
    y=df_filtrado_diario['oferta_diaria'],
    mode='lines',
    line=dict(color='royalblue', width=2),
    name='Oferta Diaria'
))

fig1.update_layout(
    title='Oferta Energética Diaria - Región Caribe',
    xaxis=dict(title='Fecha'),
    yaxis=dict(title='Oferta en GWh'),
    template='plotly_white',
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.title("Visualización de la Oferta Energética Diaria")
st.plotly_chart(fig1, use_container_width=True)

# --- Gráfico #2: Oferta diaria con eje X de 365 días (según años seleccionados) ---
df_anual = df_region_caribe[df_region_caribe['Date'].dt.year.isin(selected_years)]
df_anual['day_of_year'] = df_anual['Date'].dt.dayofyear

# Agrupar por 'day_of_year' y 'year' para representar un año ficticio
df_ficticio = df_anual.groupby(['day_of_year', 'Date']).agg({'oferta_diaria': 'sum'}).reset_index()

fig2 = go.Figure()

# Graficar la oferta diaria para cada año seleccionado
for año in selected_years:
    df_año = df_ficticio[df_ficticio['Date'].dt.year == año]
    fig2.add_trace(go.Scatter(
        x=df_año['day_of_year'],
        y=df_año['oferta_diaria'],
        mode='lines',
        name=f'Oferta Diaria {año}'
    ))

fig2.update_layout(
    title='Oferta Energética Diaria (Año Ficticio)',
    xaxis=dict(
        title='Día del Año (1-365)',
        tickmode='linear',
        dtick=30,  # Mostrar una etiqueta cada 30 días
        tickvals=list(range(1, 366, 30)),
        tickangle=0  # Mantener las etiquetas horizontales
    ),
    yaxis=dict(title='Oferta en GWh'),
    template='plotly_white',
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig2, use_container_width=True)



#Predicción de la demanda energética


# Configuración de Streamlit
st.title('Predicción de la Demanda Energética con Prophet')

# Cargar y preparar los datos
@st.cache_data
def cargar_datos():
    # Simula la carga de datos, reemplaza con tus datos reales
    data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=1000, freq='D'),
        'demanda_diaria': np.random.randint(100000, 150000, size=1000)
    })
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = cargar_datos()

# Convertir la columna de fechas al formato datetime
data['Date'] = pd.to_datetime(data['Date'])

# Preparar los datos para Prophet
data_prophet = data[['Date', 'demanda_diaria']].copy()
data_prophet.columns = ['ds', 'y']

# Crear y ajustar el modelo de Prophet
model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model_prophet.fit(data_prophet)

# Entrada de usuario para seleccionar los años de predicción
anos_prediccion = st.number_input(
    'Selecciona el número de años para predecir:',
    min_value=1,
    max_value=10,
    value=5
)

# Generar predicciones futuras
dias_prediccion = 365 * anos_prediccion
future = model_prophet.make_future_dataframe(periods=dias_prediccion)
forecast = model_prophet.predict(future)

# Entrada de usuario para seleccionar la fecha futura
fecha_futura = st.date_input(
    'Selecciona la fecha para predecir la demanda:',
    min_value=future['ds'].min().date(),
    max_value=future['ds'].max().date()
)

# Buscar la predicción para la fecha dada
prediccion = forecast[forecast['ds'] == pd.to_datetime(fecha_futura)]

# Mostrar la predicción si la fecha es válida
if not prediccion.empty:
    yhat = prediccion['yhat'].values[0]
    yhat_lower = prediccion['yhat_lower'].values[0]
    yhat_upper = prediccion['yhat_upper'].values[0]
    
    st.write(f"**Predicción para {fecha_futura}:**")
    st.write(f"- Demanda Estimada: {yhat:.2f}")
    st.write(f"- Intervalo de Confianza: [{yhat_lower:.2f}, {yhat_upper:.2f}]")
    
    # Crear la figura con Plotly
    fig = go.Figure()

    # Agregar la demanda histórica
    fig.add_trace(go.Scatter(
        x=data_prophet['ds'],
        y=data_prophet['y'],
        mode='lines',
        name='Demanda Histórica',
        line=dict(color='blue')
    ))

    # Agregar la predicción
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicción',
        line=dict(color='green')
    ))

    # Agregar el intervalo de confianza
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Límite Superior (95%)',
        line=dict(color='gray', dash='dash'),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Límite Inferior (95%)',
        line=dict(color='gray', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128, 128, 128, 0.2)'
    ))

    # Marcar la predicción específica
    fig.add_trace(go.Scatter(
        x=[pd.to_datetime(fecha_futura)],
        y=[yhat],
        mode='markers',
        name='Predicción Específica',
        marker=dict(color='red', size=10)
    ))

    # Personalización del gráfico
    fig.update_layout(
        title='Predicción de la Demanda Energética',
        xaxis_title='Fecha',
        yaxis_title='Demanda Energética',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x'
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

else:
    st.write(f"No se encontró una predicción para la fecha {fecha_futura}. Verifica si la fecha está dentro del rango de predicción.")