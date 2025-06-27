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
from io import BytesIO

# Set the title of the Streamlit application
st.title("Mapa Interactivo de las Farmacias de la Primera fase de Adjudicaciones de Andalucía")

# --------------------
# Sidebar for user inputs
st.sidebar.header("Configuración de Datos y Puntuación")

# Slider for search radius (new feature)
radio_km = st.sidebar.slider(
    "Radio (km) para sumar puntuación de municipios cercanos sin farmacia", 0, 100, 0, step=5
)

# --------------------
# Main file upload section
st.sidebar.subheader("Carga de Archivos de Datos")
uploaded_files = st.sidebar.file_uploader(
    "Sube uno o más archivos CSV con medidas", type="csv", accept_multiple_files=True
)

# Check if files are uploaded, otherwise stop the application
if not uploaded_files:
    st.warning("Por favor, sube al menos un archivo CSV para comenzar.")
    st.stop()

lista_df = []
nombres_archivos = []
territorios_file = None

# Process uploaded files
for archivo in uploaded_files:
    # Identify and separate the 'Territorios.csv' file (pharmacy data)
    if archivo.name.lower() == "territorios.csv":
        territorios_file = archivo
        continue
    try:
        # Read the CSV file, assuming semicolon delimiter
        df_temp = pd.read_csv(archivo, sep=";", na_values=["-", "", "NA"])
        # Clean column names by stripping whitespace
        df_temp.columns = df_temp.columns.str.strip()
        # Convert 'Valor' column to numeric, coercing errors to NaN
        if 'Valor' in df_temp.columns:
            df_temp['Valor'] = pd.to_numeric(df_temp['Valor'], errors='coerce')
        # Add a column to keep track of the original file name
        df_temp['__archivo__'] = archivo.name
        lista_df.append(df_temp)
        nombres_archivos.append(archivo.name)
    except Exception as e:
        st.error(f"Error al leer el archivo {archivo.name}: {e}")
        st.stop()

# Concatenate all dataframes from uploaded files into a single dataframe
df_original = pd.concat(lista_df, ignore_index=True)
st.success("Archivos cargados correctamente.")

df_farmacias = pd.DataFrame()
# Load pharmacy data if 'Territorios.csv' was uploaded
if territorios_file:
    try:
        df_farmacias = pd.read_csv(territorios_file, sep=";", na_values=["-", "", "NA"])
        df_farmacias.columns = df_farmacias.columns.str.strip()
        st.sidebar.success("Farmacias cargadas desde Territorios.csv")
    except Exception as e:
        st.sidebar.error(f"Error al leer Territorios.csv: {e}")

# --------------------
# NEW SECTION: Upload existing coordinates file to speed up geocoding
st.sidebar.subheader("Carga de Coordenadas Existentes")
archivo_coords = st.sidebar.file_uploader(
    "Sube el archivo CSV con coordenadas (ej. ieca_export_latitud_longuitud.csv)", type='csv'
)

df_coords_existentes = pd.DataFrame()
if archivo_coords is not None:
    try:
        # Load existing coordinates, assuming semicolon delimiter and comma decimal
        df_coords_raw = pd.read_csv(archivo_coords, sep=';', decimal=',', usecols=['Territorio', 'Medida', 'Valor'])
        # Pivot the raw coordinates to get Latitud and Longitud as columns
        df_coords_existentes = df_coords_raw.pivot(index='Territorio', columns='Medida', values='Valor').reset_index()
        # Ensure Latitude and Longitude are numeric
        df_coords_existentes['Latitud'] = pd.to_numeric(df_coords_existentes['Latitud'], errors='coerce')
        df_coords_existentes['Longitud'] = pd.to_numeric(df_coords_existentes['Longitud'], errors='coerce')
        st.sidebar.success("Coordenadas precargadas correctamente desde el archivo.")
    except Exception as e:
        st.sidebar.error(f"Error cargando archivo de coordenadas: {e}")

# --------------------
# Define color ranges for map markers based on score
rango_colores = [
    (0, 20, "#d73027"),   # Red
    (20, 40, "#fc8d59"),  # Orange
    (40, 60, "#fee08b"),  # Yellow
    (60, 80, "#d9ef8b"),  # Light Green
    (80, 100, "#91cf60")  # Dark Green
]

# Function to clean text by removing accents, converting to title case, and replacing non-alphanumeric with underscores
def limpiar_texto(texto):
    """
    Cleans a string by normalizing unicode characters (removing accents),
    converting to title case, and replacing non-alphanumeric characters with underscores.
    """
    texto = str(texto)
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.title()
    texto = re.sub(r'\W+', '_', texto)
    texto = texto.strip('_')
    return texto

# Function to combine 'Medida' with other extra columns into a single string
def combinar_medida_y_extras(row, extras):
    """
    Combines the 'Medida' column value with values from specified 'extras' columns
    into a single underscore-separated string, after cleaning each part.
    """
    parts = [str(row['Medida']).strip()]
    for col in extras:
        val = str(row[col]).strip()
        # Exclude empty or 'NaN' like strings
        if val and val.lower() not in ['nan', 'none', 'na', '']:
            parts.append(val)
    clean_parts = [limpiar_texto(p) for p in parts]
    return "_".join(clean_parts)

# Function to normalize indicator names for consistent lookup
def normaliza_nombre_indicador(nombre):
    """
    Normalizes an indicator name by removing accents, converting to lowercase,
    and replacing non-alphanumeric characters with underscores.
    """
    nombre = str(nombre)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
    nombre = nombre.lower()
    nombre = re.sub(r'[^a-z0-9_]', '_', nombre)
    nombre = re.sub(r'_+', '_', nombre)
    return nombre.strip('_')

# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

# Function to obtain coordinates for territories, using existing ones if available
@st.cache_data(show_spinner=False)
def obtener_coordenadas(territorios, df_coords_existentes):
    """
    Obtains geographical coordinates for a list of territories.
    Prioritizes existing coordinates from df_coords_existentes,
    otherwise uses Nominatim for geocoding with rate limiting.
    """
    geolocator = Nominatim(user_agent="andalucia-mapa")
    # Rate limiter to avoid hitting API limits
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=2)

    resultados = []
    coords_dict = {}

    # Populate dictionary with existing coordinates for faster lookup
    if not df_coords_existentes.empty:
        for _, row in df_coords_existentes.iterrows():
            t = row['Territorio'].strip()
            lat = row['Latitud']
            lon = row['Longitud']
            if pd.notna(lat) and pd.notna(lon):
                coords_dict[t] = (lat, lon)

    # Iterate through territories to get coordinates
    for lugar in territorios:
        lugar_clean = lugar.strip()
        # Use existing coordinates if available
        if lugar_clean in coords_dict:
            lat, lon = coords_dict[lugar_clean]
            resultados.append((lugar_clean, lat, lon))
            continue

        # If not in existing, try geocoding
        try:
            location = geocode(f"{lugar_clean}, Andalucía, España", timeout=10)
            if location:
                resultados.append((lugar_clean, location.latitude, location.longitude))
            else:
                resultados.append((lugar_clean, None, None))
        except Exception as e:
            # Handle geocoding errors, append None for coordinates
            st.warning(f"No se pudieron obtener coordenadas para {lugar_clean}: {e}")
            resultados.append((lugar_clean, None, None))

    return pd.DataFrame(resultados, columns=["Territorio", "Latitud", "Longitud"])

# --------------------
# Load Weights from CSV (New Section)
st.sidebar.subheader("Cargar/Guardar Pesos")
uploaded_weights_file = st.sidebar.file_uploader(
    "Sube un archivo CSV con pesos guardados", type="csv", key="weights_uploader"
)

loaded_pesos_dict = {}
if uploaded_weights_file is not None:
    try:
        df_loaded_pesos = pd.read_csv(uploaded_weights_file, sep=';')
        if 'Indicador' in df_loaded_pesos.columns and 'Peso' in df_loaded_pesos.columns:
            loaded_pesos_dict = pd.Series(df_loaded_pesos.Peso.values, index=df_loaded_pesos.Indicador).to_dict()
            st.sidebar.success("Pesos cargados correctamente.")
        else:
            st.sidebar.error("El archivo de pesos debe contener las columnas 'Indicador' y 'Peso'.")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el archivo de pesos: {e}")

# Create a selection tree for weights per data file in the sidebar
st.sidebar.subheader("Pesos por archivo de datos")
pesos = {}          # Dictionary to store weights for each indicator
medidas_originales = {} # Dictionary to store original indicator names

# Dynamically create sliders for each indicator from each uploaded file
for archivo in nombres_archivos:
    with st.sidebar.expander(f"⚙️ {archivo}", expanded=False):
        df_archivo = df_original[df_original['__archivo__'] == archivo]
        columnas_basicas = {'Territorio', 'Medida', 'Valor', '__archivo__'}
        # Identify extra columns that are not basic data fields
        columnas_extra = [col for col in df_archivo.columns if col not in columnas_basicas]
        
        # Get unique combined indicators for the current file
        # This will combine the original 'Medida' with other extra columns to form the slider label
        indicadores_combinados = df_archivo.apply(
            lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1
        ).unique()
        
        # Create a slider for each unique combined indicator
        for indicador_completo in sorted(indicadores_combinados):
            clave_norm = normaliza_nombre_indicador(indicador_completo)
            
            # Initialize slider value with loaded weights, if available, otherwise default to 1.0
            initial_peso = loaded_pesos_dict.get(clave_norm, 1.0)

            peso = st.slider(f"{indicador_completo}", -5.0, 5.0, initial_peso, 0.1, key=f"{archivo}-{clave_norm}")
            pesos[clave_norm] = peso
            medidas_originales[clave_norm] = indicador_completo # Store original name for display

# Determine extra columns (outside the loop to be used globally for df_original)
columnas_basicas = {'Territorio', 'Medida', 'Valor'}
columnas_extra = [col for col in df_original.columns if col not in columnas_basicas and col != '__archivo__']

# Generar medida combinada para cada fila, esto actualiza la columna 'Medida' en df_original
# para que contenga el nombre completo del indicador (ej. "Poblacion_Rural")
df_original['Medida'] = df_original.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1)

# Function to normalize municipality names for consistent matching
def normalizar_nombre_municipio(nombre):
    """
    Normalizes a municipality name by removing accents, converting to lowercase,
    and removing non-alphanumeric characters (except spaces).
    """
    nombre = str(nombre)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
    nombre = nombre.lower()
    nombre = re.sub(r'[^a-z0-9 ]', '', nombre)
    return nombre.strip()

# Function to prepare and process the data, including pivoting, geocoding, and scoring
@st.cache_data
def preparar_datos(df_original, pesos, df_coords_existentes, df_farmacias, radio_km):
    """
    Prepares the data by pivoting, normalizing columns, identifying
    municipalities with/without pharmacies, geocoding them, calculating scores,
    and extending scores based on nearby municipalities.
    """
    # DEBUG: Show initial df_original columns
    st.write(f"DEBUG: df_original columns: {df_original.columns.tolist()}")
    # DEBUG: Show sample of df_original['Medida'] after global combination
    st.write(f"DEBUG: Sample of df_original['Medida'] (combined indicator names): {df_original['Medida'].head().tolist()}")

    columnas_basicas = {'Territorio', 'Medida', 'Valor'}
    columnas_extra_preparar = [col for col in df_original.columns if col not in columnas_basicas and col != '__archivo__']

    # Pivot the original dataframe to have measures as columns
    df_pivot = df_original.pivot_table(
        index="Territorio",
        columns="Medida", # This 'Medida' is now the already combined string
        values="Valor",
        aggfunc="first"
    ).reset_index()

    # DEBUG: Show df_pivot columns before renaming
    st.write(f"DEBUG: df_pivot columns BEFORE normalizing: {df_pivot.columns.tolist()}")

    # Normalizar nombres de columnas para que coincidan con las claves de 'pesos'
    col_map = {col: normaliza_nombre_indicador(col) if col != 'Territorio' else col for col in df_pivot.columns}
    df_pivot = df_pivot.rename(columns=col_map)
    # DEBUG: Show df_pivot columns after renaming
    #st.write(f"DEBUG: df_pivot columns AFTER normalizing: {df_pivot.columns.tolist()}")

    # DEBUG: Show keys in 'pesos' dictionary
    #st.write(f"DEBUG: Keys in 'pesos' dictionary (from sliders): {list(pesos.keys())}")

    # Add a normalized territory name column for matching with pharmacy data
    df_pivot["Territorio_normalizado"] = df_pivot["Territorio"].apply(normalizar_nombre_municipio)

    municipios_con_farmacia = set()
    if not df_farmacias.empty:
        if 'Territorio' in df_farmacias.columns:
            df_farmacias["Territorio_normalizado"] = df_farmacias["Territorio"].apply(normalizar_nombre_municipio)
            municipios_con_farmacia = set(df_farmacias["Territorio_normalizado"])

    # Separate dataframes for municipalities with and without pharmacies
    df_con_farmacia = df_pivot[df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()
    df_sin_farmacia = df_pivot[~df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()

    # Obtain coordinates for both sets of municipalities
    with st.spinner("Obteniendo coordenadas de municipios con farmacia..."):
        df_coordenadas_con = obtener_coordenadas(df_con_farmacia["Territorio"], df_coords_existentes)
    df_con_farmacia = df_con_farmacia.merge(df_coordenadas_con, on="Territorio", how="left")

    with st.spinner("Obteniendo coordenadas de municipios sin farmacia..."):
        df_coordenadas_sin = obtener_coordenadas(df_sin_farmacia["Territorio"], df_coords_existentes)
    df_sin_farmacia = df_sin_farmacia.merge(df_coordenadas_sin, on="Territorio", how="left")

    # Calculate the base score for municipalities with pharmacies
    # Sum of (indicator value * its weight)
    # DEBUG: Check columns actually being used for scoring
    cols_for_scoring_con = [col for col in pesos if col in df_con_farmacia.columns]
    st.write(f"DEBUG: Columns in df_con_farmacia being used for scoring: {cols_for_scoring_con}")

    df_con_farmacia['Puntuación'] = sum(
        df_con_farmacia[col].fillna(0) * pesos.get(col, 0)
        for col in pesos if col in df_con_farmacia.columns
    )
    # DEBUG: Show sample of 'Puntuación' column
    st.write(f"DEBUG: Sample of 'Puntuación' column (first 5 rows, municipalities with pharmacy): {df_con_farmacia['Puntuación'].head().tolist()}")

    # Calculate the base score for municipalities without pharmacies
    cols_for_scoring_sin = [col for col in pesos if col in df_sin_farmacia.columns]
    st.write(f"DEBUG: Columns in df_sin_farmacia being used for scoring: {cols_for_scoring_sin}")
    df_sin_farmacia['Puntuación'] = sum(
        df_sin_farmacia[col].fillna(0) * pesos.get(col, 0)
        for col in pesos if col in df_sin_farmacia.columns
    )
    st.write(f"DEBUG: Sample of 'Puntuación' column (first 5 rows, municipalities without pharmacy): {df_sin_farmacia['Puntuación'].head().tolist()}")


    # Initialize extended score and nearby municipalities sum
    df_con_farmacia['PuntuaciónExtendida'] = df_con_farmacia['Puntuación']
    df_con_farmacia['SumaMunicipiosCercanos'] = 0.0

    # If a search radius is set, calculate extended scores
    if radio_km > 0:
        # For each municipality with a pharmacy, sum the scores of nearby municipalities without pharmacies
        for idx, row in df_con_farmacia.iterrows():
            lat1, lon1 = row['Latitud'], row['Longitud']
            if pd.isna(lat1) or pd.isna(lon1):
                continue
            suma_extra = 0
            # Iterate through municipalities without pharmacies
            for _, r2 in df_sin_farmacia.iterrows():
                lat2, lon2 = r2['Latitud'], r2['Longitud']
                if pd.isna(lat2) or pd.isna(lon2):
                    continue
                distancia = haversine(lat1, lon1, lat2, lon2)
                if distancia <= radio_km:
                    suma_extra += r2['Puntuación'] # Add the score of the nearby municipality
            df_con_farmacia.at[idx, 'SumaMunicipiosCercanos'] = suma_extra
            df_con_farmacia.at[idx, 'PuntuaciónExtendida'] = row['Puntuación'] + suma_extra

    st.write(f"DEBUG: Sample of 'PuntuaciónExtendida' column (first 5 rows): {df_con_farmacia['PuntuaciónExtendida'].head().tolist()}")

    return df_con_farmacia, df_sin_farmacia

# Prepare the dataframes using the cached function
df_municipios_farmacias, df_municipios_sin = preparar_datos(
    df_original, pesos, df_coords_existentes, df_farmacias, radio_km
)

# -------------------
# Display ranking table and allow selection
df_ordenado = df_municipios_farmacias.sort_values('PuntuaciónExtendida', ascending=False)
st.subheader("Ranking de municipios con farmacia ordenados por puntuación total (incluye suma municipios sin farmacia cercanos)")

# Check if df_ordenado is empty before creating the selectbox
if not df_ordenado.empty:
    territorio_seleccionado = st.selectbox(
        "Selecciona un municipio del ranking para centrar el mapa:",
        options=df_ordenado['Territorio'].tolist()
    )
else:
    territorio_seleccionado = None
    st.info("No hay municipios con farmacia para mostrar en el ranking.")


st.dataframe(df_ordenado[[
    'Territorio', 'Puntuación', 'SumaMunicipiosCercanos', 'PuntuaciónExtendida'
]].round(2), use_container_width=True)

# Display detailed breakdown for the selected territory
if territorio_seleccionado:
    st.subheader(f"Detalle de puntuación para: {territorio_seleccionado}")

    # Filter original dataframe for the selected territory
    df_territorio = df_original[df_original["Territorio"] == territorio_seleccionado]

    if df_territorio.empty:
        st.warning("No hay datos detallados para este territorio.")
        df_desglose = pd.DataFrame() # Initialize an empty dataframe
    else:
        st.write(f"Número de filas originales para {territorio_seleccionado}: ", len(df_original[df_original["Territorio"] == territorio_seleccionado]))
        desglose = []

        for _, row in df_territorio.iterrows():
            clave_norm = normaliza_nombre_indicador(row["Medida"]) # 'Medida' here is already the combined string

            valor = row["Valor"]
            peso = pesos.get(clave_norm, 1.0)  # Use default weight of 1.0 if not found
            contribucion = valor * peso if pd.notna(valor) else None

            # Find the original full name of the indicator for display
            original_display_name = medidas_originales.get(clave_norm, row["Medida"])

            desglose.append({
                "Indicador": original_display_name,
                "Valor": round(valor, 2) if pd.notna(valor) else "N/A",
                "Peso": round(peso, 2),
                "Contribución": round(contribucion, 2) if contribucion is not None else "—"
            })

        df_desglose = pd.DataFrame(desglose)
        st.dataframe(df_desglose, use_container_width=True, height=600)

    st.write(f"Nº de indicadores en el desglose: {len(df_desglose)}")

    # Download full breakdown CSV
    csv_buffer_desglose = BytesIO()
    df_desglose.to_csv(csv_buffer_desglose, index=False)
    csv_buffer_desglose.seek(0)

    st.download_button(
        label="📥 Descargar desglose completo en CSV",
        data=csv_buffer_desglose,
        file_name=f"desglose_{territorio_seleccionado}.csv",
        mime="text/csv"
    )

# -------------------
# Folium Map
st.subheader("Mapa Interactivo de Municipios")

# Default coordinates for map center
lat_centro, lon_centro = 37.4, -5.9
zoom_nivel = 7

# Adjust map center and zoom if a territory is selected
if territorio_seleccionado and territorio_seleccionado != "(Ninguno)": # Added check for None
    fila_sel = df_ordenado[df_ordenado['Territorio'] == territorio_seleccionado]
    if not fila_sel.empty and pd.notna(fila_sel.iloc[0]['Latitud']) and pd.notna(fila_sel.iloc[0]['Longitud']):
        lat_centro = fila_sel.iloc[0]['Latitud']
        lon_centro = fila_sel.iloc[0]['Longitud']
        zoom_nivel = 11 # Zoom in closer to the selected territory

# Create the Folium map
m = folium.Map(location=[lat_centro, lon_centro], zoom_start=zoom_nivel)
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each municipality with a pharmacy
for idx, row in df_ordenado.iterrows():
    lat, lon = row['Latitud'], row['Longitud']
    # Skip if coordinates are missing
    if pd.isna(lat) or pd.isna(lon):
        continue

    color = "#777777"  # Default gray color for markers
    puntuacion = row['PuntuaciónExtendida']
    # Determine marker color based on extended score range
    for (minv, maxv, col) in rango_colores:
        if minv <= puntuacion < maxv:
            color = col
            break
    # If score is 100 or above, assign the last color
    if puntuacion >= rango_colores[-1][1]:
        color = rango_colores[-1][2]

    # Create HTML content for the popup
    popup_html = f"""
    <b>{row['Territorio']}</b><br>
    Puntuación propia: {row['Puntuación']:.2f}<br>
    Suma municipios cercanos sin farmacia (≤ {radio_km} km): {row['SumaMunicipiosCercanos']:.2f}<br>
    <b>Total combinado:</b> {row['PuntuaciónExtendida']:.2f}
    """

    # Add a circle marker to the map
    folium.CircleMarker(
        location=(lat, lon),
        radius=7, # Size of the circle marker
        popup=folium.Popup(popup_html, max_width=300), # Popup on click
        color=color,        # Border color
        fill=True,          # Fill the circle
        fill_color=color,   # Fill color
        fill_opacity=0.7,   # Fill opacity
    ).add_to(marker_cluster)

# Add a fullscreen button to the map
Fullscreen().add_to(m)

# Display the Folium map in Streamlit
st_data = st_folium(m, width=1200, height=700, returned_objects=["last_clicked"])

# -------------------
# Plotly Bar Chart for total combined scores
st.subheader("Gráfico de puntuación total combinada")
fig = px.bar(
    df_ordenado,
    x='Territorio',
    y='PuntuaciónExtendida',
    color='PuntuaciónExtendida', # Color bars based on their score
    color_continuous_scale='Viridis', # Color scale
    labels={'PuntuaciónExtendida': 'Puntuación Total'}, # Axis label
    height=400
)
fig.update_layout(xaxis_tickangle=-45) # Rotate x-axis labels for readability
st.plotly_chart(fig, use_container_width=True) # Display Plotly chart

# -------------------
# Export complete processed data
st.subheader("📥 Descargar datos procesados")

# Concatenate municipalities with and without pharmacies into a single dataframe for export
df_export = pd.concat([df_municipios_farmacias, df_municipios_sin], ignore_index=True)

# Reorder columns if necessary (optional, for better readability in CSV)
cols_first = ["Territorio", "Latitud", "Longitud", "Puntuación", "SumaMunicipiosCercanos", "PuntuaciónExtendida"]
# Ensure all columns from original df_export are included, with desired ones first
cols_others = [col for col in df_export.columns if col not in cols_first and col != "Territorio_normalizado"]
df_export = df_export[cols_first + sorted(cols_others)] # Sort other columns alphabetically

# Convert the dataframe to CSV format
csv_data = df_export.to_csv(index=False, sep=";", encoding="utf-8").encode("utf-8")

# Download button for the complete processed CSV
st.download_button(
    label="📥 Descargar CSV con todos los municipios",
    data=csv_data,
    file_name="todos_los_municipios.csv",
    mime="text/csv"
)

# Sidebar button to clear Streamlit cache
if st.sidebar.button("🧹 Limpiar caché de datos"):
    st.cache_data.clear() # Clear all cached data
    st.experimental_rerun() # Rerun the app to reflect changes

# --------------------
# Save Weights to CSV (New Section)
st.sidebar.subheader("Guardar Pesos Actuales")
if st.sidebar.button("💾 Guardar Pesos a CSV"):
    if pesos:
        df_pesos_guardar = pd.DataFrame(pesos.items(), columns=['Indicador', 'Peso'])
        # Use medidas_originales to map back to original display names for clarity in the saved CSV
        # Ensure 'Indicador_Original' column is created before using it for reordering
        df_pesos_guardar['Indicador_Original'] = df_pesos_guardar['Indicador'].map(medidas_originales)
        
        # Reorder columns for readability in the saved CSV
        df_pesos_guardar = df_pesos_guardar[['Indicador_Original', 'Indicador', 'Peso']]
        
        csv_buffer_pesos = BytesIO()
        df_pesos_guardar.to_csv(csv_buffer_pesos, index=False, sep=';', encoding='utf-8')
        csv_buffer_pesos.seek(0)
        
        st.sidebar.download_button(
            label="Descargar pesos_guardados.csv",
            data=csv_buffer_pesos,
            file_name="pesos_guardados.csv",
            mime="text/csv",
            key="download_weights_button"
        )
        st.sidebar.success("Pesos listos para descargar.")
    else:
        st.sidebar.warning("No hay pesos para guardar. Carga archivos de datos primero.")


# -------------------
# Version information in the sidebar
st.sidebar.subheader("Version 1.1.0")
