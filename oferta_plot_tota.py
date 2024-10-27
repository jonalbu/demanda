import pandas as pd
import streamlit as st
import plotly.express as px

# Cargar el archivo de datos
df_oferta = pd.read_csv('C:/Users/jonal/OneDrive/Documentos/12_Bootcamp_Talento_Tech/Proyecto/Streamlit/oferta_recursos.txt', sep='\t')

# Asegúrate de que la columna 'Date' esté en formato datetime
df_oferta['Date'] = pd.to_datetime(df_oferta['Date'])

# Suma de la oferta por horas (diaria), sin eliminar los valores nulos
df_oferta['oferta_diaria'] = df_oferta.sum(axis=1, skipna=False, numeric_only=True)

# Definir los recursos específicos de la "Costa Caribe"
recursos_costa_caribe = ["CTG1", "CTG2", "CTG3", "EPFV", "GE32", "MATA", "PRG1", "PRG2", "TBST"]

# Filtrar el DataFrame por los recursos de la "Costa Caribe"
df_costa_caribe = df_oferta[df_oferta['Values_code'].isin(recursos_costa_caribe)]

# --- Sidebar para selección de fuentes energéticas y rango de fechas ---
st.sidebar.header("Configuración de Filtro")

# Obtener los tipos de 'Values_Type' únicos
values_type_unicos = df_costa_caribe['Values_Type'].unique()

# Widget para seleccionar múltiples tipos de 'Values_Type'
selected_values_type = st.sidebar.multiselect(
    "Selecciona las fuentes energéticas",
    options=values_type_unicos,
    default=values_type_unicos[:1]  # Selecciona el primer tipo por defecto
)

# Filtrar el DataFrame según los tipos de 'Values_Type' seleccionados
df_filtrado = df_costa_caribe[df_costa_caribe['Values_Type'].isin(selected_values_type)]

# Rango de fechas mínimo y máximo
min_date = df_filtrado['Date'].min()
max_date = df_filtrado['Date'].max()

# Widget para seleccionar el rango de fechas
selected_range = st.sidebar.date_input(
    "Selecciona el rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filtrar el DataFrame según el rango de fechas seleccionado
df_filtrado = df_filtrado[
    (df_filtrado['Date'] >= pd.to_datetime(selected_range[0])) & 
    (df_filtrado['Date'] <= pd.to_datetime(selected_range[1]))
]

# Agrupar por 'Values_Type' y sumar la oferta diaria (incluyendo valores nulos)
df_agrupado = df_filtrado.groupby('Values_Type')['oferta_diaria'].sum(min_count=1).reset_index()

# Crear el gráfico de torta con Plotly
fig = px.pie(
    df_agrupado,
    values='oferta_diaria',
    names='Values_Type',
    title="Distribución de la Oferta por Fuentes Energéticas - Costa Caribe",
    hole=0.4  # Añadir un agujero para convertirlo en un gráfico de dona
)

# Agregar los porcentajes dentro de las secciones de la torta
fig.update_traces(textposition='inside', textinfo='percent+label')

# Mostrar el gráfico en Streamlit
st.title("Visualización de la Distribución de la Oferta por Fuentes Energéticas - Costa Caribe")
st.plotly_chart(fig, use_container_width=True)
