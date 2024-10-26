
import datetime as dt  
import mysql.connector  # Importar el conector de MySQL
import pandas as pd
#import plost
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
import os


try:
    # Establecer la conexión
    conn = mysql.connector.connect('mysql', type='sql')
    #st.write("Conexión exitosa a la base de datos.")

    st.set_page_config(layout = 'wide', initial_sidebar_state='expanded')
    
    with open ('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.sidebar.header('Demanda energética')
    
    
    # Consulta a la base de datos (modifica la tabla según la que necesites)
    query = "SELECT MONTH(fecha) AS Mes, SUM(valor) AS GWh FROM demanda d GROUP BY Mes ORDER BY Mes"
    data = pd.read_sql_query(query, conn)

    # Mostrar los datos obtenidos
    st.title("Demanda Energética")
    #st.subheader("Datos demanda Energética por mes")
    #st.write(data)

    # Gráfico de barras: Demanda total energética por mes
    st.subheader("Demanda total energética por mes:")
    fig1, ax1 = plt.subplots()
    ax1.bar(data['Mes'], data['GWh'])
    ax1.set_xlabel('Mes')
    ax1.set_ylabel('Demanda (GWh)')
    ax1.set_title('Demanda Energética por Mes')
    st.pyplot(fig1)

    # Consulta 2: Demanda por regiones
    query_2 = """
    SELECT r.descripcion AS Region, SUM(d.valor) AS Demanda_Total_GWh 
    FROM demanda d 
    JOIN region r ON d.codigo_reg = r.codigo_region 
    GROUP BY r.descripcion
    ORDER BY Demanda_Total_GWh asc;
    """
    data_2 = pd.read_sql_query(query_2, conn)

    # Mostrar los datos obtenidos
    #st.subheader("Datos de la demanda por regiones:")
    #st.write(data_2)

    # Gráfico de barras: Demanda total por regiones
    #st.subheader("Demanda total por regiones:")
    #st.bar_chart(data_2.set_index('Region')['Demanda_Total_GWh'].sort_values(ascending=False))
    
    st.subheader("Demanda total por regiones:")
    data_2_sorted = data_2.sort_values(by='Demanda_Total_GWh', ascending=False)
    st.bar_chart(data_2_sorted.set_index('Region')['Demanda_Total_GWh'])
    
    # Consulta 3: Demanda por horas
    
    query_3 = """
    SELECT
    DATE_FORMAT(fecha, '%Y-%m-%d') as fecha,
    hour(fecha) as hora,
    avg(valor) AS GWh
    FROM demanda d
    GROUP BY fecha, hora
    ORDER BY fecha, hora
    """
    
    df = pd.read_sql_query(query_3, conn)
    
    # Mostrar los datos obtenidos
    st.subheader("Datos de la demanda por hora:")
    
    # Mostrar los datos obtenidos
    #st.write(df)
    
    # Gráfico de calor (Demanda de energía por hora)
    
       
    fig = px.density_heatmap(df, x="fecha", y="hora", z='GWh',
                             color_continuous_scale='YlGnBu',
                                 title="Horarios de mayor demanda")
    st.plotly_chart(fig)
        
    

except mysql.connector.Error as err:
    st.write(f"Error al conectar con la base de datos: {err}")
finally:
    # Cerrar la conexión si se estableció correctamente
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        st.write("Conexión cerrada.")
