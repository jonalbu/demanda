import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

##Empieza el Código




# Configuración de la página de Streamlit
st.set_page_config(layout='wide', initial_sidebar_state='expanded')


st.sidebar.image("Logo/logo2.jpg", use_column_width=True)

with open('style2.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    

st.sidebar.header('Demanda Energética')

# Cargar el archivo de datos
df_demanda = pd.read_csv('Data/demanda_exportada.txt', sep='\t')

# Asegurar que la columna 'Date' esté en formato datetime
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
    yaxis=dict(title='Demanda en Wh'),
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
    yaxis=dict(title='Demanda en KWh'),
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
        yaxis=dict(title='Demanda en KWh'),
        template='plotly_dark',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.title("Visualización de la Demanda Energética Diaria Superpuesta")
    st.plotly_chart(fig_superpuesto, use_container_width=True)
else:
    st.write("Por favor, selecciona al menos un año para visualizar la demanda diaria.")


# OFERTA ENERGÉTICA
    
# Cargar el DataFrame
df_oferta2 = pd.read_csv('Data/oferta_recursos.txt', sep='\t')
df_oferta = df_oferta2.dropna()

# Asegúrate de que la columna 'Date' esté en formato datetime
df_oferta['Date'] = pd.to_datetime(df_oferta['Date'])

# Suma de la oferta diaria y eliminación de valores nulos
df_oferta['oferta_diaria'] = df_oferta.sum(axis=1, skipna=True, numeric_only=True)
df_oferta = df_oferta.dropna(subset=['oferta_diaria'])

# Sidebar para la selección de rango de fechas, tipos de fuente energética, regiones, departamentos y años
with st.sidebar:
    st.header("Oferta Energética")
    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona el rango de fechas",
        [min(df_oferta['Date']), max(df_oferta['Date'])],
        min_value=min(df_oferta['Date']),
        max_value=max(df_oferta['Date'])
    )
    values_types = st.multiselect("Selecciona los tipos de fuente energética", df_oferta['Values_Type'].unique())
    regiones = st.multiselect("Selecciona las regiones", df_oferta['Región'].unique())
    departamentos = st.multiselect("Selecciona los departamentos", df_oferta['Departamento'].unique())
    years = st.multiselect("Selecciona el/los año(s)", df_oferta['Date'].dt.year.unique())

# Filtrar el DataFrame según los criterios seleccionados
df_filtrado = df_oferta[
    (df_oferta['Values_Type'].isin(values_types) | (not values_types)) &
    (df_oferta['Región'].isin(regiones) | (not regiones)) &
    (df_oferta['Departamento'].isin(departamentos) | (not departamentos)) &
    (df_oferta['Date'] >= pd.to_datetime(fecha_inicio)) &
    (df_oferta['Date'] <= pd.to_datetime(fecha_fin)) &
    (df_oferta['Date'].dt.year.isin(years) | (not years))
]

# Gráfico #1: Oferta por meses (líneas)
df_mensual = df_filtrado.groupby([pd.Grouper(key='Date', freq='M'), 'Región', 'Departamento'])['oferta_diaria'].sum().reset_index()
fig1 = px.line(df_mensual, x='Date', y='oferta_diaria', color='Región',
               line_group='Departamento',
               labels={'Date': 'Fecha', 'oferta_diaria': 'Oferta (KWh)', 'Región': 'Región'},
               title='Oferta Mensual en Colombia por Región y Departamento')

# Gráfico #2: Oferta por días (líneas)
df_diaria = df_filtrado.groupby([pd.Grouper(key='Date', freq='D'), 'Región', 'Departamento'])['oferta_diaria'].sum().reset_index()
fig2 = px.line(df_diaria, x='Date', y='oferta_diaria', color='Región',
               line_group='Departamento',
               labels={'Date': 'Fecha', 'oferta_diaria': 'Oferta (KWh)', 'Región': 'Región'},
               title='Oferta Diaria en Colombia por Región y Departamento')

# Gráfico de torta #1: Aportes de tipo de fuente de energía
df_pie = df_filtrado.groupby('Values_Type')['oferta_diaria'].sum().reset_index()
fig_pie = px.pie(df_pie, values='oferta_diaria', names='Values_Type',
                 title='Aportes de Tipo de Fuente de Energía',
                 labels={'Values_Type': 'Tipo de Fuente', 'oferta_diaria': 'Oferta Total (KWh)'})

# Gráfico de torta #2: Distribución de Values_Name filtrada por regiones y departamentos seleccionados
df_pie_Values_Name = df_filtrado.groupby(['Values_Name', 'Región', 'Departamento'])['oferta_diaria'].sum().reset_index()

# Filtrar por las regiones y departamentos seleccionados
df_pie_Values_Name = df_pie_Values_Name[
    (df_pie_Values_Name['Región'].isin(regiones) | (not regiones)) &
    (df_pie_Values_Name['Departamento'].isin(departamentos) | (not departamentos))
]

fig_pie_Values_Name = px.pie(df_pie_Values_Name, values='oferta_diaria', names='Values_Name',
                             title='Distribución Planta Energética',
                             labels={'Values_Name': 'Values Name', 'oferta_diaria': 'Oferta Total (KWh)'})

# Diseño de columnas para gráficos en paralelo
col1, col2 = st.columns([2, 1])

with col1:
    # Mostrar el gráfico de oferta por meses
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Mostrar el gráfico de torta de tipo de fuente de energía
    st.plotly_chart(fig_pie, use_container_width=True)

# Diseño de columnas para el gráfico de oferta diaria y el gráfico de torta de Values_Name
col3, col4 = st.columns([2, 1])

with col3:
    # Mostrar el gráfico de oferta diaria
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    # Mostrar el gráfico de torta de Values_Name debajo de la primera torta
    st.plotly_chart(fig_pie_Values_Name, use_container_width=True)

# Gráfico #3: Sumatoria de oferta diaria por año seleccionado
df_anual = df_filtrado.copy()
df_anual['Dia_del_año'] = df_anual['Date'].dt.dayofyear

# Agrupar por día del año y calcular la sumatoria de oferta diaria para los años seleccionados
df_anual_sum = df_anual.groupby(['Dia_del_año', df_anual['Date'].dt.year])['oferta_diaria'].sum().reset_index()
fig3 = px.line(df_anual_sum, x='Dia_del_año', y='oferta_diaria', color='Date',
               labels={'Dia_del_año': 'Día del Año', 'oferta_diaria': 'Oferta Total (KWh)', 'Date': 'Año'},
               title='Sumatoria de la Oferta Diaria por Año')

# Mostrar el tercer gráfico después de los dos primeros
st.plotly_chart(fig3, use_container_width=True)




#Predicción de la demanda energética

# Cargar los datos (modifica la ruta según tu directorio)
demanda = pd.read_csv('Data/demanda_diaria_regresion.txt', sep='\t')
oferta = pd.read_csv('Data/oferta_diaria_regresion.txt', sep='\t')

# Combinar los datos
data = pd.merge(demanda, oferta, on='Date')

# Crear nuevas características
data['diferencia_demanda'] = data['demanda_diaria'].diff().fillna(0)
data['media_movil_demanda'] = data['demanda_diaria'].rolling(window=3).mean().fillna(data['demanda_diaria'].mean())
data['lag_demanda'] = data['demanda_diaria'].shift(1).fillna(data['demanda_diaria'].mean())

# Características y variable objetivo
X = data[['oferta_diaria', 'diferencia_demanda', 'media_movil_demanda', 'lag_demanda']]
y = data['demanda_diaria']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline con escalado y modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', GradientBoostingRegressor())
])

# Definir los hiperparámetros a buscar en GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.1]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Predicciones con el mejor modelo
y_pred = best_model.predict(X_test)

# Evaluación del modelo optimizado
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Interfaz de Streamlit
st.title('Modelo de Predicción de Demanda Energética')

# Mostrar las métricas de evaluación
st.subheader('Métricas de Evaluación')
#st.write(f"Mejores Hiperparámetros: {grid_search.best_params_}")
st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"R2: {r2:.2f}")

# Implementar la predicción con Prophet
data['Date'] = pd.to_datetime(data['Date'])
data_prophet = data[['Date', 'demanda_diaria']].copy()
data_prophet.columns = ['ds', 'y']

# Crear y ajustar el modelo de Prophet
model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model_prophet.fit(data_prophet)

# Inputs para la selección del número de años y la fecha específica
anos_prediccion = st.number_input('Selecciona el número de años para predecir:', min_value=1, max_value=10, value=5, step=1)
fecha_futura = st.date_input('Selecciona la fecha para predecir la demanda:', value=pd.to_datetime('2024-10-27'))

# Convertir la fecha de predicción a Timestamp
fecha_futura = pd.to_datetime(fecha_futura)

# Opción para mostrar u ocultar intervalos de confianza
mostrar_intervalo = st.checkbox('Mostrar intervalos de confianza', value=True)

# Verificar si la fecha futura es válida
ultima_fecha = data['Date'].max()

if fecha_futura <= ultima_fecha:
    st.error(f"La fecha seleccionada debe ser posterior a {ultima_fecha.date()}.")
else:
    # Función para predecir la demanda
    def predecir_demanda(fecha_futura):
        dias_prediccion = (fecha_futura - ultima_fecha).days
        future = model_prophet.make_future_dataframe(periods=dias_prediccion)
        forecast = model_prophet.predict(future)

        prediccion = forecast[forecast['ds'] == fecha_futura]

        if not prediccion.empty:
            yhat = prediccion['yhat'].values[0]

            st.subheader(f'Predicción para {fecha_futura.date()}:')
            st.write(f"Demanda Estimada: {yhat:.2f}")

            # Graficar la predicción completa usando Plotly
            fig = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Fecha', 'yhat': 'Demanda Estimada'},
                          title='Predicción de la Demanda Energética')
            fig.add_scatter(x=data_prophet['ds'], y=data_prophet['y'], mode='lines', name='Demanda Histórica')
            #fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicción', line=dict(color='green'))

            # Agregar intervalos de confianza si se selecciona
            if mostrar_intervalo:
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Intervalo Superior', line=dict(dash='dot'))
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Intervalo Inferior', line=dict(dash='dot'))

            # Marcar la predicción específica con un punto
            fig.add_scatter(x=[fecha_futura], y=[yhat], mode='markers', name='Predicción Específica', marker=dict(color='red', size=10))

            st.plotly_chart(fig)

    # Llamar a la función de predicción con la fecha seleccionada
    predecir_demanda(fecha_futura)




#Predicción de la oferta energética


# Cargar los datos (demanda y oferta)
demanda = pd.read_csv('Data/demanda_diaria_regresion.txt', sep='\t')
oferta = pd.read_csv('Data/oferta_diaria_regresion.txt', sep='\t')

# Convertir la columna de fecha a formato datetime
demanda['Date'] = pd.to_datetime(demanda['Date'])
oferta['Date'] = pd.to_datetime(oferta['Date'])

# Combinar los datos
data = pd.merge(demanda, oferta, on='Date')

# Preparar los datos para el modelo
data.set_index('Date', inplace=True)
oferta_data = data['oferta_diaria']

# Configuración de Streamlit
st.title('Predicción de la Oferta Energética')

# Colocación de las selecciones en una sola fila
col1, col2 = st.columns(2)

with col1:
    # Input numérico para seleccionar el número de años a predecir con botones + y -
    n_years = st.number_input(
        'Selecciona el número de años para predecir:',
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

with col2:
    # Entrada para la fecha final de la predicción
    end_date = st.date_input(
        'Selecciona la fecha final de la predicción:',
        value=pd.to_datetime(oferta_data.index[-1] + pd.DateOffset(years=1))
    )

# Asegúrate de que end_date sea de tipo datetime
end_date = pd.to_datetime(end_date)

# Espacio adicional para separación visual
st.markdown('---')

# Calcular el número de días para la predicción desde la última fecha de datos hasta la fecha seleccionada
ultima_fecha = oferta_data.index[-1]

# Asegúrate de que ultima_fecha sea de tipo datetime
ultima_fecha = pd.to_datetime(ultima_fecha)

# Calcular el número de días para la predicción
n_periods = (end_date - ultima_fecha).days

# Ajustar el modelo de suavización exponencial (Holt-Winters)
model = ExponentialSmoothing(oferta_data, trend='add', seasonal='add', seasonal_periods=365)
fit = model.fit()

# Predicción a futuro (en días) desde la última fecha de la serie histórica
forecast = fit.forecast(steps=n_periods)

# Calcular el intervalo de confianza del 95%
alpha = 0.05
ci_upper = forecast + 1.96 * fit.resid.std()
ci_lower = forecast - 1.96 * fit.resid.std()

# Crear un DataFrame para la predicción y los intervalos de confianza
forecast_index = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=n_periods, freq='D')
df_forecast = pd.DataFrame({'Fecha': forecast_index, 'Predicción': forecast, 
                            'IC_Superior': ci_upper, 'IC_Inferior': ci_lower})

# Obtener el valor estimado en la fecha final seleccionada
end_pred_value = df_forecast[df_forecast['Fecha'] == end_date]['Predicción'].iloc[0]

# Mostrar la predicción para la fecha seleccionada
st.markdown(f"### Predicción para {end_date.strftime('%Y-%m-%d')}:")
st.write(f"Oferta Estimada: {end_pred_value:.2f}")

# Graficar oferta histórica y predicción
fig = go.Figure()

# Oferta histórica
fig.add_trace(go.Scatter(
    x=oferta_data.index, 
    y=oferta_data, 
    mode='lines',
    name='Oferta Histórica',
    line=dict(color='blue')
))

# Predicción
fig.add_trace(go.Scatter(
    x=df_forecast['Fecha'], 
    y=df_forecast['Predicción'], 
    mode='lines',
    name='Predicción',
    line=dict(color='green')
))

# Intervalo de confianza (siempre visible)
fig.add_trace(go.Scatter(
    x=pd.concat([df_forecast['Fecha'], df_forecast['Fecha'][::-1]]),
    y=pd.concat([df_forecast['IC_Superior'], df_forecast['IC_Inferior'][::-1]]),
    fill='toself',
    fillcolor='rgba(128, 128, 128, 0.2)',
    line=dict(color='rgba(128, 128, 128, 0)'),
    name='Intervalos de Confianza (95%)'
))

# Agregar un punto rojo en la fecha final de la predicción con la oferta estimada
fig.add_trace(go.Scatter(
    x=[end_date],
    y=[end_pred_value],
    mode='markers+text',
    marker=dict(color='red', size=10),
    name='Predicción Específica',
    text=[f'{end_pred_value:.2f}'],
    textposition='top center'
))

# Configuración de la gráfica
fig.update_layout(
    title='Predicción de la Oferta Energética',
    xaxis_title='Fecha',
    yaxis_title='Oferta Estimada',
    legend=dict(x=0, y=1, traceorder='normal'),
    xaxis=dict(tickangle=45),
    template='plotly_white'
)

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)
