import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import unicodedata
import re
from math import radians, cos, sin, asin, sqrt

st.title("Mapa Interactivo de Pueblos de Andalucía")

# --------------------
# Slider para radio de búsqueda (nuevo)
radio_km = st.sidebar.slider(
    "Radio (km) para sumar puntuación de municipios cercanos sin farmacia", 0, 100, 0, step=5
)

# --------------------
# Subida de archivos principales
uploaded_files = st.sidebar.file_uploader("Sube uno o más archivos CSV con medidas", type="csv", accept_multiple_files=True)
if not uploaded_files:
    st.warning("Por favor, sube al menos un archivo CSV para comenzar.")
    st.stop()

lista_df = []
nombres_archivos = []
territorios_file = None
for archivo in uploaded_files:
    if archivo.name.lower() == "territorios.csv":
        territorios_file = archivo
        continue
    try:
        df_temp = pd.read_csv(archivo, sep=";", na_values=["-", "", "NA"])
        df_temp.columns = df_temp.columns.str.strip()
        if 'Valor' in df_temp.columns:
            df_temp['Valor'] = pd.to_numeric(df_temp['Valor'], errors='coerce')
        df_temp['__archivo__'] = archivo.name
        lista_df.append(df_temp)
        nombres_archivos.append(archivo.name)
    except Exception as e:
        st.error(f"Error al leer el archivo {archivo.name}: {e}")
        st.stop()

df_original = pd.concat(lista_df, ignore_index=True)
st.success("Archivos cargados correctamente.")

df_farmacias = pd.DataFrame()
if territorios_file:
    try:
        df_farmacias = pd.read_csv(territorios_file, sep=";", na_values=["-", "", "NA"])
        df_farmacias.columns = df_farmacias.columns.str.strip()
        st.sidebar.success("Farmacias cargadas desde Territorios.csv")
    except Exception as e:
        st.sidebar.error(f"Error al leer Territorios.csv: {e}")

# --------------------
# NUEVA PARTE: Subida archivo coordenadas para acelerar geocodificación
archivo_coords = st.sidebar.file_uploader("Sube el archivo CSV con coordenadas (ieca_export_latitud_longuitud.csv)", type='csv')

df_coords_existentes = pd.DataFrame()
if archivo_coords is not None:
    try:
        df_coords_raw = pd.read_csv(archivo_coords, sep=';', decimal=',', usecols=['Territorio', 'Medida', 'Valor'])
        df_coords_existentes = df_coords_raw.pivot(index='Territorio', columns='Medida', values='Valor').reset_index()
        df_coords_existentes['Latitud'] = pd.to_numeric(df_coords_existentes['Latitud'], errors='coerce')
        df_coords_existentes['Longitud'] = pd.to_numeric(df_coords_existentes['Longitud'], errors='coerce')
        st.sidebar.success("Coordenadas precargadas correctamente desde el archivo.")
    except Exception as e:
        st.sidebar.error(f"Error cargando archivo de coordenadas: {e}")

# --------------------
rango_colores = [
    (0, 20, "#d73027"),
    (20, 40, "#fc8d59"),
    (40, 60, "#fee08b"),
    (60, 80, "#d9ef8b"),
    (80, 100, "#91cf60")
]

def limpiar_texto(texto):
    texto = str(texto)
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.title()
    texto = re.sub(r'\W+', '_', texto)
    texto = texto.strip('_')
    return texto

def combinar_medida_y_extras(row, extras):
    parts = [str(row['Medida']).strip()]
    for col in extras:
        val = str(row[col]).strip()
        if val and val.lower() not in ['nan', 'none', 'na', '']:
            parts.append(val)
    clean_parts = [limpiar_texto(p) for p in parts]
    return "_".join(clean_parts)

def normaliza_nombre_indicador(nombre):
    nombre = str(nombre)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
    nombre = nombre.lower()
    nombre = re.sub(r'[^a-z0-9_]', '_', nombre)
    nombre = re.sub(r'_+', '_', nombre)
    return nombre.strip('_')

def haversine(lat1, lon1, lat2, lon2):
    # Distancia entre dos puntos (lat, lon) en km
    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

@st.cache_data(show_spinner=False)
def obtener_coordenadas(territorios, df_coords_existentes):
    geolocator = Nominatim(user_agent="andalucia-mapa")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=2)

    resultados = []
    coords_dict = {}

    if not df_coords_existentes.empty:
        for _, row in df_coords_existentes.iterrows():
            t = row['Territorio'].strip()
            lat = row['Latitud']
            lon = row['Longitud']
            if pd.notna(lat) and pd.notna(lon):
                coords_dict[t] = (lat, lon)

    for lugar in territorios:
        lugar_clean = lugar.strip()
        if lugar_clean in coords_dict:
            lat, lon = coords_dict[lugar_clean]
            resultados.append((lugar_clean, lat, lon))
            continue

        try:
            location = geocode(f"{lugar_clean}, Andalucía, España", timeout=10)
            if location:
                resultados.append((lugar_clean, location.latitude, location.longitude))
            else:
                resultados.append((lugar_clean, None, None))
        except Exception:
            resultados.append((lugar_clean, None, None))

    return pd.DataFrame(resultados, columns=["Territorio", "Latitud", "Longitud"])

# Crear árbol de selección de pesos por archivo
st.sidebar.subheader("Pesos por archivo de datos")
pesos = {}
medidas_originales = {}
for archivo in nombres_archivos:
    with st.sidebar.expander(f"⚙️ {archivo}", expanded=False):
        indicadores_archivo = df_original[df_original['__archivo__'] == archivo]['Medida'].unique()
        for Medida in indicadores_archivo:
            clave_norm = normaliza_nombre_indicador(Medida)
            peso = st.slider(f"{Medida}", 0.0, 5.0, 1.0, 0.1, key=f"{archivo}-{Medida}")
            pesos[clave_norm] = peso
            medidas_originales[clave_norm] = Medida

# Al preparar los datos, normaliza también las columnas del pivot
def normalizar_nombre_municipio(nombre):
    nombre = str(nombre)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
    nombre = nombre.lower()
    nombre = re.sub(r'[^a-z0-9 ]', '', nombre)
    return nombre.strip()

@st.cache_data
def preparar_datos(df_original, pesos, df_coords_existentes, df_farmacias, radio_km):
    columnas_basicas = {'Territorio', 'Medida', 'Valor'}
    columnas_extra = [col for col in df_original.columns if col not in columnas_basicas and col != '__archivo__']
    df_original['Medida'] = df_original.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1)
    df_pivot = df_original.pivot_table(
        index="Territorio",
        columns="Medida",
        values="Valor",
        aggfunc="first"
    ).reset_index()

    # Normalizar nombres de columnas
    col_map = {col: normaliza_nombre_indicador(col) if col != 'Territorio' else col for col in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=col_map)

    # Añadir columna de nombre normalizado para hacer cruce con farmacias
    df_pivot["Territorio_normalizado"] = df_pivot["Territorio"].apply(normalizar_nombre_municipio)

    municipios_con_farmacia = set()
    if not df_farmacias.empty:
        if 'Territorio' in df_farmacias.columns:
            df_farmacias["Territorio_normalizado"] = df_farmacias["Territorio"].apply(normalizar_nombre_municipio)
            municipios_con_farmacia = set(df_farmacias["Territorio_normalizado"])

    # df municipios con farmacia
    df_con_farmacia = df_pivot[df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()
    # df municipios sin farmacia
    df_sin_farmacia = df_pivot[~df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()

    # Obtener coordenadas
    df_coordenadas_con = obtener_coordenadas(df_con_farmacia["Territorio"], df_coords_existentes)
    df_con_farmacia = df_con_farmacia.merge(df_coordenadas_con, on="Territorio", how="left")

    df_coordenadas_sin = obtener_coordenadas(df_sin_farmacia["Territorio"], df_coords_existentes)
    df_sin_farmacia = df_sin_farmacia.merge(df_coordenadas_sin, on="Territorio", how="left")

    # Calcular puntuación solo para municipios con farmacia
    df_con_farmacia['Puntuación'] = sum(df_con_farmacia[col].fillna(0) * pesos.get(col, 0) for col in pesos if col in df_con_farmacia.columns)
    # Calcular puntuación para municipios sin farmacia
    df_sin_farmacia['Puntuación'] = sum(df_sin_farmacia[col].fillna(0) * pesos.get(col, 0) for col in pesos if col in df_sin_farmacia.columns)

    # Añadir columna para puntuación extendida (inicial = puntuación propia)
    df_con_farmacia['PuntuaciónExtendida'] = df_con_farmacia['Puntuación']
    df_con_farmacia['SumaMunicipiosCercanos'] = 0.0

    if radio_km > 0:
        # Por cada municipio con farmacia, sumar puntuación de municipios sin farmacia dentro del radio
        for idx, row in df_con_farmacia.iterrows():
            lat1, lon1 = row['Latitud'], row['Longitud']
            if pd.isna(lat1) or pd.isna(lon1):
                continue
            suma_extra = 0
            for _, r2 in df_sin_farmacia.iterrows():
                lat2, lon2 = r2['Latitud'], r2['Longitud']
                if pd.isna(lat2) or pd.isna(lon2):
                    continue
                distancia = haversine(lat1, lon1, lat2, lon2)
                if distancia <= radio_km:
                    suma_extra += r2['Puntuación']
            df_con_farmacia.at[idx, 'SumaMunicipiosCercanos'] = suma_extra
            df_con_farmacia.at[idx, 'PuntuaciónExtendida'] = row['Puntuación'] + suma_extra

    return df_con_farmacia, df_sin_farmacia

df_municipios_farmacias, df_municipios_sin = preparar_datos(df_original, pesos, df_coords_existentes, df_farmacias, radio_km)

# -------------------
# Filtrado por provincia si se quiere
provincias_disponibles = df_municipios_farmacias['Territorio'].apply(lambda x: x.split()[0] if " " in x else "").unique()
provincias_disponibles = [p for p in provincias_disponibles if p]

provincia_seleccionada = st.sidebar.selectbox("Filtra por provincia (opcional)", options=["Todas"] + list(provincias_disponibles))

if provincia_seleccionada != "Todas":
    df_municipios_farmacias = df_municipios_farmacias[df_municipios_farmacias['Territorio'].str.startswith(provincia_seleccionada)]
    df_municipios_sin = df_municipios_sin[df_municipios_sin['Territorio'].str.startswith(provincia_seleccionada)]

# -------------------
# Mostrar ranking ordenado por puntuación extendida descendente
df_ordenado = df_municipios_farmacias.sort_values('PuntuaciónExtendida', ascending=False)
st.subheader("Ranking de municipios con farmacia ordenados por puntuación total (incluye suma municipios sin farmacia cercanos)")

st.dataframe(df_ordenado[[
    'Territorio', 'Puntuación', 'SumaMunicipiosCercanos', 'PuntuaciónExtendida'
]].round(2))

# -------------------
# Mapa con folium
m = folium.Map(location=[37.4, -5.9], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

for idx, row in df_ordenado.iterrows():
    lat, lon = row['Latitud'], row['Longitud']
    if pd.isna(lat) or pd.isna(lon):
        continue

    color = "#777777"  # gris default
    puntuacion = row['PuntuaciónExtendida']
    for (minv, maxv, col) in rango_colores:
        if minv <= puntuacion < maxv:
            color = col
            break

    popup_html = f"""
    <b>{row['Territorio']}</b><br>
    Puntuación propia: {row['Puntuación']:.2f}<br>
    Suma municipios cercanos sin farmacia (≤ {radio_km} km): {row['SumaMunicipiosCercanos']:.2f}<br>
    <b>Total combinado:</b> {row['PuntuaciónExtendida']:.2f}
    """

    folium.CircleMarker(
        location=(lat, lon),
        radius=7,
        popup=folium.Popup(popup_html, max_width=300),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
    ).add_to(marker_cluster)

Fullscreen().add_to(m)
st_data = st_folium(m, width=700, height=500)

# -------------------
# Gráfico Plotly
st.subheader("Gráfico de puntuación total combinada")
fig = px.bar(
    df_ordenado,
    x='Territorio',
    y='PuntuaciónExtendida',
    color='PuntuaciónExtendida',
    color_continuous_scale='Viridis',
    labels={'PuntuaciónExtendida': 'Puntuación Total'},
    height=400
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)
