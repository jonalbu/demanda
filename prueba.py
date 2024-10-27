import streamlit as st
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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