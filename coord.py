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
import numpy as np
from types import SimpleNamespace

# Set the title of the Streamlit application
st.title("Mapa Interactivo de las Farmacias de la Primera fase de Adjudicaciones de Andalucía")

# --------------------
# Navigation tabs
tab1, tab2 = st.tabs(["🗺️ Mapa y Ranking", "📊 Comparación de Municipios"])

# --------------------
# Sidebar for user inputs
st.sidebar.header("Configuración de Datos y Puntuación")

# --------------------
# Lista de archivos CSV en GitHub
uploaded_files = [
"Territorios.csv",
"ieca_export_alquileres.csv",
"ieca_export_att_especializada.csv",
"ieca_export_att_primaria.csv",
"ieca_export_bancos.csv",
"ieca_export_centro_educativos.csv",
"ieca_export_centros_asistenciales.csv",
"ieca_export_centros_sociales.csv",
"ieca_export_contratos_registrados.csv",
"ieca_export_corbertura.csv",
"ieca_export_emigraciones_edad_sexo.csv",
"ieca_export_establec_turisticos.csv",
"ieca_export_establecimientos.csv",
"ieca_export_explot_ganaderas.csv",
"ieca_export_fcia_poblacion.csv",
"ieca_export_inmigraciones_edad_sexo.csv",
"ieca_export_inmigración_extranjeros.csv",
"ieca_export_instalaciones_deportivas.csv",
"ieca_export_latitud_longuitud.csv",
"ieca_export_poblacion_edad_nac.csv",
"ieca_export_renta.csv",
"singular_pob_sexo.csv"
]

# Convertimos a objetos con atributo .name
uploaded_files = [SimpleNamespace(name=item) for item in uploaded_files]

lista_df = []
nombres_archivos = []
territorios_file = None
df_coords_existentes = pd.DataFrame()

for archivo in uploaded_files:
    if archivo.name.lower() == "territorios.csv":
            territorios_file = archivo
            continue
    if archivo.name.lower() == "ieca_export_latitud_longuitud.csv":
            # --------------------
            # Nuevo código: cargar automáticamente desde GitHub
            try:
                df_coords_raw = pd.read_csv(archivo.name, sep=';', decimal=',', usecols=['Territorio', 'Medida', 'Valor'])
                df_coords_existentes = df_coords_raw.pivot(index='Territorio', columns='Medida', values='Valor').reset_index()
                df_coords_existentes['Latitud'] = pd.to_numeric(df_coords_existentes['Latitud'], errors='coerce')
                df_coords_existentes['Longitud'] = pd.to_numeric(df_coords_existentes['Longitud'], errors='coerce')
                st.sidebar.success("Coordenadas cargadas automáticamente desde ieca_export_latitud_longuitud.csv")
            except Exception as e:
                st.sidebar.error(f"Error cargando ieca_export_latitud_longuitud.csv: {e}")
            continue
    
            # --------------------
            # Código antiguo (comentado): pedía subir el CSV manualmente
            # uploaded_file = st.file_uploader("Sube ieca_export_latitud_longuitud.csv", type="csv")
            # if uploaded_file is not None:
            #     df_coords_raw = pd.read_csv(uploaded_file, sep=';', decimal=',', usecols=['Territorio', 'Medida', 'Valor'])
            #     df_coords_existentes = df_coords_raw.pivot(index='Territorio', columns='Medida', values='Valor').reset_index()
            #     df_coords_existentes['Latitud'] = pd.to_numeric(df_coords_existentes['Latitud'], errors='coerce')
            #     df_coords_existentes['Longitud'] = pd.to_numeric(df_coords_existentes['Longitud'], errors='coerce')
            #     st.sidebar.success("Coordenadas cargadas desde archivo subido")

    try:
            # Nuevo: usamos archivo.name
            df_temp = pd.read_csv(archivo.name, sep=";", na_values=["-", "", "NA"])
            df_temp.columns = df_temp.columns.str.strip()
            if 'Valor' in df_temp.columns:
                df_temp['Valor'] = pd.to_numeric(df_temp['Valor'], errors='coerce')
            df_temp['__archivo__'] = archivo.name
            lista_df.append(df_temp)
            nombres_archivos.append(archivo.name)
    except Exception as e:
            st.error(f"Error al leer el archivo {archivo.name}: {e}")
            st.stop()

# Concatenamos todo en un único DataFrame
df_original = pd.concat(lista_df, ignore_index=True)
st.success("Archivos cargados correctamente.Espere")

# --------------------
# Territorios.csv

df_farmacias = pd.DataFrame()
if territorios_file:
    try:
            # Nuevo: usamos territorios_file.name
            df_farmacias = pd.read_csv(territorios_file.name, sep=";", na_values=["-", "", "NA"])
            df_farmacias.columns = df_farmacias.columns.str.strip()
            if 'Singular' in df_farmacias.columns:
                df_farmacias['Nombre_Mostrar'] = df_farmacias['Singular'].fillna(df_farmacias['Territorio'])
            else:
                df_farmacias['Nombre_Mostrar'] = df_farmacias['Territorio']
            st.sidebar.success("Farmacias cargadas desde Territorios.csv")
    except Exception as e:
            st.sidebar.error(f"Error al leer Territorios.csv: {e}")

    # --------------------
    # Código antiguo (comentado)
    # df_farmacias = pd.read_csv(territorios_file, sep=";", na_values=["-", "", "NA"])

# --------------------
# Guardamos coordenadas en sesión
if 'df_coords' not in st.session_state:
    st.session_state.df_coords = df_coords_existentes
    st.session_state.df_coords_original = df_coords_existentes.copy()
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

    nuevos_territorios = [t for t in territorios if t.strip() not in coords_dict]
    progress_bar = st.sidebar.progress(0, text="Geocodificando...")
    total_to_geocode = len(nuevos_territorios)

    for i, lugar in enumerate(nuevos_territorios):
            lugar_clean = lugar.strip()
            try:
                location = geocode(f"{lugar_clean}, Andalucía, España", timeout=10)
                if location:
                    resultados.append((lugar_clean, location.latitude, location.longitude))
                else:
                    resultados.append((lugar_clean, None, None))
            except Exception as e:
                st.sidebar.warning(f"No se pudieron obtener coordenadas para {lugar_clean}: {e}")
                resultados.append((lugar_clean, None, None))
            progress_bar.progress((i + 1) / total_to_geocode, text=f"Geocodificando {lugar_clean}...")

    progress_bar.empty()
    df_nuevas_coords = pd.DataFrame(resultados, columns=["Territorio", "Latitud", "Longitud"])
    return pd.concat([df_coords_existentes, df_nuevas_coords], ignore_index=True)

if st.sidebar.button("Geolocalizar Municipios Faltantes"):
    municipios_unicos = df_original["Territorio"].unique()
    with st.spinner("Geolocalizando... esto puede tardar un poco la primera vez."):
            st.session_state.df_coords = obtener_coordenadas(municipios_unicos, st.session_state.df_coords_original)
    st.sidebar.success("Geolocalización completada.")

if st.session_state.df_coords.empty:
    st.info("Carga un archivo de coordenadas o usa el botón 'Geolocalizar Municipios Faltantes' para continuar.")
    st.stop()

# --------------------
# Helper Functions (unmodified)
rango_colores = [
    (0, 20, "#d73027"), (20, 40, "#fc8d59"), (40, 60, "#fee08b"),
    (60, 80, "#d9ef8b"), (80, 100, "#91cf60")
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

def normalizar_nombre_municipio(nombre):
    nombre = str(nombre)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
    nombre = nombre.lower()
    nombre = re.sub(r'[^a-z0-9 ]', '', nombre)
    return nombre.strip()

# --------------------
# Load Weights from CSV
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

# --- INICIO DEL FORMULARIO ---
st.sidebar.subheader("Ajuste de Pesos y Parámetros")

radio_km = st.sidebar.slider(
"Radio (km) para sumar puntuación de municipios cercanos sin farmacia", 0, 100, 0, step=1
)

pesos = {}
medidas_originales = {}

# --- Primero renderizamos los expansores con sliders y botones fuera del form ---
st.sidebar.markdown("### Configuración de pesos por archivo")

for archivo in nombres_archivos:
    with st.sidebar.expander(f"⚙️ {archivo}", expanded=False):
        df_archivo = df_original[df_original['__archivo__'] == archivo]
        columnas_basicas = {'Territorio', 'Medida', 'Valor', '__archivo__'}
        columnas_extra = [col for col in df_archivo.columns if col not in columnas_basicas]
        indicadores_combinados = df_archivo.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1).unique()

        # Campo para valor global y botón fuera del form (permitido)
        col1, col2 = st.columns([0.7, 0.3])
        # Establecer valor por defecto en 0 para explot_ganaderas, 1.0 para el resto
        valor_por_defecto = 0.0 if "explot_ganaderas" in archivo else 1.0
        valor_global = col1.number_input(
            f"Valor global para {archivo}", -5.0, 5.0, valor_por_defecto, 0.1, key=f"global_val_{archivo}"
        )
        if col2.button("Aplicar", key=f"aplicar_{archivo}"):
            for indicador_completo in sorted(indicadores_combinados):
                clave_norm = normaliza_nombre_indicador(indicador_completo)
                st.session_state[f"{archivo}-{clave_norm}"] = valor_global
            st.rerun()

        # Sliders individuales
        for indicador_completo in sorted(indicadores_combinados):
            clave_norm = normaliza_nombre_indicador(indicador_completo)
            # Establecer peso por defecto en 0 para explot_ganaderas, 1.0 para el resto
            peso_por_defecto = 0.0 if "explot_ganaderas" in archivo else 1.0
            initial_peso = st.session_state.get(f"{archivo}-{clave_norm}", loaded_pesos_dict.get(clave_norm, peso_por_defecto))
            peso = st.slider(f"{indicador_completo}", -5.0, 5.0, initial_peso, 0.1, key=f"{archivo}-{clave_norm}")
            pesos[clave_norm] = peso
            medidas_originales[clave_norm] = indicador_completo

# --- Formulario solo para recalcular ---
with st.sidebar.form("config_form"):
    recalcular_button = st.form_submit_button("Aplicar Cambios y Recalcular")
# --- FIN DEL FORMULARIO ---

# El resto del código solo se ejecuta si se envía el formulario
# o si se carga la página por primera vez.

# --------------------
# TAB 1: Mapa y Ranking
with tab1:
    columnas_basicas = {'Territorio', 'Medida', 'Valor'}
    columnas_extra = [col for col in df_original.columns if col not in columnas_basicas and col != '__archivo__']
    df_original['Medida'] = df_original.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1)

    @st.cache_data
    def preparar_datos_base(df_original, df_coords, df_farmacias):
        df_pivot = df_original.pivot_table(
            index="Territorio", columns="Medida", values="Valor", aggfunc="first"
        ).reset_index()
        col_map = {col: normaliza_nombre_indicador(col) if col != 'Territorio' else col for col in df_pivot.columns}
        df_pivot = df_pivot.rename(columns=col_map)
        df_pivot["Territorio_normalizado"] = df_pivot["Territorio"].apply(normalizar_nombre_municipio)
        municipios_con_farmacia = set()
        df_farmacias_factores = pd.DataFrame()
        if not df_farmacias.empty:
            if 'Territorio' in df_farmacias.columns and 'Factor' in df_farmacias.columns:
                df_farmacias["Territorio_normalizado"] = df_farmacias["Territorio"].apply(normalizar_nombre_municipio)
                municipios_con_farmacia = set(df_farmacias["Territorio_normalizado"])
                df_farmacias_factores = df_farmacias[["Territorio_normalizado", "Factor", "Nombre_Mostrar"]].copy()

        df_con_farmacia_base = df_pivot[df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()
        df_sin_farmacia_base = df_pivot[~df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()

        if not df_farmacias_factores.empty:
            df_con_farmacia_base = pd.merge(df_con_farmacia_base, df_farmacias_factores, on="Territorio_normalizado", how="left")
            df_con_farmacia_base['Factor'] = df_con_farmacia_base['Factor'].fillna(1.0)
        else:
            df_con_farmacia_base['Factor'] = 1.0
            df_con_farmacia_base['Nombre_Mostrar'] = df_con_farmacia_base['Territorio']

        df_con_farmacia_base = pd.merge(df_con_farmacia_base, df_coords, on="Territorio", how="left")
        df_sin_farmacia_base = pd.merge(df_sin_farmacia_base, df_coords, on="Territorio", how="left")
        return df_con_farmacia_base, df_sin_farmacia_base

    def calcular_puntuaciones(df_con_farmacia_base, df_sin_farmacia_base, pesos, radio_km):
        df_con_farmacia = df_con_farmacia_base.copy()
        df_sin_farmacia = df_sin_farmacia_base.copy()
        df_con_farmacia['Puntuación'] = sum(
            df_con_farmacia[col].fillna(0) * pesos.get(col, 0)
            for col in pesos if col in df_con_farmacia.columns
        )
        df_sin_farmacia['Puntuación'] = sum(
            df_sin_farmacia[col].fillna(0) * pesos.get(col, 0)
            for col in pesos if col in df_sin_farmacia.columns
        )
        df_con_farmacia['PuntuaciónFinal'] = df_con_farmacia['Puntuación'] * df_con_farmacia['Factor']
        df_con_farmacia['PuntuaciónExtendida'] = df_con_farmacia['PuntuaciónFinal']
        df_con_farmacia['SumaMunicipiosCercanos'] = 0.0

        if radio_km > 0 and not df_sin_farmacia.empty:
            df_con_farmacia_valid = df_con_farmacia.dropna(subset=['Latitud', 'Longitud'])
            df_sin_farmacia_valid = df_sin_farmacia.dropna(subset=['Latitud', 'Longitud'])
            if not df_con_farmacia_valid.empty:
                lat_con_rad = np.radians(df_con_farmacia_valid['Latitud'])
                lon_con_rad = np.radians(df_con_farmacia_valid['Longitud'])
                lat_sin_rad = np.radians(df_sin_farmacia_valid['Latitud'])
                lon_sin_rad = np.radians(df_sin_farmacia_valid['Longitud'])
                dlon = lon_sin_rad.values[:, None] - lon_con_rad.values
                dlat = lat_sin_rad.values[:, None] - lat_con_rad.values
                a = np.sin(dlat / 2.0)**2 + np.cos(lat_sin_rad.values[:, None]) * np.cos(lat_con_rad.values) * np.sin(dlon / 2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distancias = 6371 * c
                indices_dentro_radio = np.where(distancias <= radio_km)
                puntuaciones_cercanas = pd.Series(
                    df_sin_farmacia_valid['Puntuación'].values[indices_dentro_radio[0]],
                    index=df_con_farmacia_valid.index[indices_dentro_radio[1]]
                ).groupby(level=0).sum()
                df_con_farmacia.loc[puntuaciones_cercanas.index, 'SumaMunicipiosCercanos'] = puntuaciones_cercanas
                df_con_farmacia['PuntuaciónExtendida'] = df_con_farmacia['PuntuaciónFinal'] + df_con_farmacia['SumaMunicipiosCercanos']
        return df_con_farmacia, df_sin_farmacia

# --- FLUJO PRINCIPAL ---
    df_con_farmacia_base, df_sin_farmacia_base = preparar_datos_base(
    df_original, st.session_state.df_coords, df_farmacias
    )

    df_municipios_farmacias, df_municipios_sin = calcular_puntuaciones(
    df_con_farmacia_base, df_sin_farmacia_base, pesos, radio_km
    )

# -------------------
# Display ranking table and allow selection
    df_ordenado = df_municipios_farmacias.sort_values('PuntuaciónExtendida', ascending=False).reset_index(drop=True)
    df_ordenado.index += 1  # Índice 1-based

    st.subheader("Ranking de municipios con farmacia ordenados por puntuación total")

    if not df_ordenado.empty:
    territorio_seleccionado = st.selectbox(
            "Selecciona un municipio del ranking para centrar el mapa:",
            options=df_ordenado['Nombre_Mostrar'].tolist()
    )
    else:
    territorio_seleccionado = None
    st.info("No hay municipios con farmacia para mostrar en el ranking.")

    st.dataframe(
    df_ordenado.reset_index().rename(columns={"index": "Ranking"})[
            ['Ranking', 'Nombre_Mostrar', 'Puntuación', 'Factor', 'PuntuaciónFinal', 'SumaMunicipiosCercanos', 'PuntuaciónExtendida']
    ].round(2),
    use_container_width=True
    )


# Display detailed breakdown for the selected territory
    if territorio_seleccionado:
    st.subheader(f"Detalle de puntuación para: {territorio_seleccionado}")

    fila_farmacia = df_municipios_farmacias[df_municipios_farmacias["Nombre_Mostrar"] == territorio_seleccionado]
    territorio_original_para_desglose = fila_farmacia.iloc[0]['Territorio'] if not fila_farmacia.empty else None

    if territorio_original_para_desglose:
            df_territorio = df_original[df_original["Territorio"] == territorio_original_para_desglose]
    else:
            df_territorio = pd.DataFrame()

    if df_territorio.empty:
            st.warning("No hay datos detallados para este territorio.")
            df_desglose = pd.DataFrame()
    else:
            st.write(f"Número de indicadores para {territorio_seleccionado}: ", len(df_territorio))
            desglose = []
            puntuacion_base = 0
            for _, row in df_territorio.iterrows():
                clave_norm = normaliza_nombre_indicador(row["Medida"])
                valor = row["Valor"]
                peso = pesos.get(clave_norm, 1.0)
                contribucion = valor * peso if pd.notna(valor) else 0
                puntuacion_base += contribucion
                original_display_name = medidas_originales.get(clave_norm, row["Medida"])
                desglose.append({
                    "Indicador": original_display_name,
                    "Valor": round(valor, 2) if pd.notna(valor) else "N/A",
                    "Peso": round(peso, 2),
                    "Contribución": round(contribucion, 2) if contribucion is not None else "—"
                })
            df_desglose = pd.DataFrame(desglose)
            st.dataframe(df_desglose, use_container_width=True, height=600)

    if not fila_farmacia.empty:
            factor_valor = fila_farmacia.iloc[0]['Factor']
            puntuacion_final = fila_farmacia.iloc[0]['PuntuaciónFinal']
            st.write(f"**Puntuación base (suma de contribuciones):** {puntuacion_base:.2f}")
            st.write(f"**Factor aplicado:** {factor_valor:.2f}")
            st.write(f"**Puntuación con factor:** {puntuacion_final:.2f}")

    csv_buffer_desglose = BytesIO()
    df_desglose.to_csv(csv_buffer_desglose, index=False)
    csv_buffer_desglose.seek(0)
    st.download_button(
            label="📥 Descargar desglose completo en CSV",
            file_name=f"desglose_{territorio_seleccionado}.csv",
            data=csv_buffer_desglose,
            mime="text/csv"
    )

# -------------------
# Folium Map
    st.subheader("Mapa Interactivo de Municipios")

    lat_centro, lon_centro = 37.4, -5.9
    zoom_nivel = 7

    if territorio_seleccionado and not df_ordenado.empty:
    fila_sel = df_ordenado[df_ordenado['Nombre_Mostrar'] == territorio_seleccionado]
    if not fila_sel.empty and pd.notna(fila_sel.iloc[0]['Latitud']) and pd.notna(fila_sel.iloc[0]['Longitud']):
            lat_centro = fila_sel.iloc[0]['Latitud']
            lon_centro = fila_sel.iloc[0]['Longitud']
            zoom_nivel = 11

    m = folium.Map(location=[lat_centro, lon_centro], zoom_start=zoom_nivel)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in df_ordenado.iterrows():
    lat, lon = row['Latitud'], row['Longitud']
    if pd.isna(lat) or pd.isna(lon):
            continue

    color = "#777777"
    puntuacion = row['PuntuaciónExtendida']
    for (minv, maxv, col) in rango_colores:
            if minv <= puntuacion < maxv:
                color = col
                break
    if puntuacion >= rango_colores[-1][1]:
            color = rango_colores[-1][2]

    popup_html = f"""
    <b>{row['Nombre_Mostrar']}</b><br>
    Puntuación base: {row['Puntuación']:.2f}<br>
    Factor: {row['Factor']:.2f}<br>
    Puntuación con factor: {row['PuntuaciónFinal']:.2f}<br>
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
    st_data = st_folium(m, width=1200, height=700, returned_objects=["last_clicked"])

# -------------------
# Plotly Bar Chart
    st.subheader("Gráfico de puntuación total combinada")
    fig = px.bar(
    df_ordenado,
    x='Nombre_Mostrar',
    y='PuntuaciónExtendida',
    color='PuntuaciónExtendida',
    color_continuous_scale='Viridis',
    labels={'PuntuaciónExtendida': 'Puntuación Total', 'Nombre_Mostrar': 'Nombre Entidad'},
    height=400
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# -------------------
# Export complete processed data
    st.subheader("📥 Descargar datos procesados")
    df_export = pd.concat([df_municipios_farmacias, df_municipios_sin], ignore_index=True)
    cols_first = ["Nombre_Mostrar", "Territorio", "Latitud", "Longitud", "Puntuación", "Factor", "PuntuaciónFinal", "SumaMunicipiosCercanos", "PuntuaciónExtendida"]
    cols_others = [col for col in df_export.columns if col not in cols_first and col != "Territorio_normalizado"]
    df_export = df_export[cols_first + sorted(cols_others)]
    csv_data = df_export.to_csv(index=False, sep=";", encoding="utf-8").encode("utf-8")
    st.download_button(
    label="📥 Descargar CSV con todos los municipios",
    data=csv_data,
    file_name="todos_los_municipios.csv",
    mime="text/csv"
    )

# Sidebar button to clear Streamlit cache
    if st.sidebar.button("🧹 Limpiar caché de datos"):
    st.cache_data.clear()
    if 'df_coords' in st.session_state:
            del st.session_state.df_coords
    if 'df_coords_original' in st.session_state:
            del st.session_state.df_coords_original
    st.experimental_rerun()

# --------------------
# --------------------
# Guardar Pesos Actuales
    st.sidebar.subheader("Guardar Pesos Actuales")
    if pesos:
    df_pesos_guardar = pd.DataFrame(pesos.items(), columns=['Indicador', 'Peso'])
    df_pesos_guardar['Indicador_Original'] = df_pesos_guardar['Indicador'].map(medidas_originales)
    df_pesos_guardar = df_pesos_guardar[['Indicador_Original', 'Indicador', 'Peso']]
    csv_buffer_pesos = BytesIO()
    df_pesos_guardar.to_csv(csv_buffer_pesos, index=False, sep=';', encoding='utf-8')
    csv_buffer_pesos.seek(0)
    st.sidebar.download_button(
            label="💾 Descargar configuración actual de pesos",
            data=csv_buffer_pesos,
            file_name="pesos_guardados.csv",
            mime="text/csv",
            key="download_weights_button"
    )
    else:
    st.sidebar.warning("No hay pesos para guardar. Carga archivos de datos primero.")

# --------------------
# TAB 2: Comparación de Municipios
with tab2:
st.header("📊 Comparación de Municipios")

# Verificar que tenemos datos cargados
if 'df_municipios_farmacias' not in locals() or df_municipios_farmacias.empty:
    st.warning("⚠️ Primero debes cargar los datos y calcular las puntuaciones en la pestaña 'Mapa y Ranking'.")
    st.info("Ve a la primera pestaña, configura los pesos y presiona 'Aplicar Cambios y Recalcular'.")
else:
    # Obtener lista de municipios disponibles
    municipios_disponibles = df_municipios_farmacias['Nombre_Mostrar'].tolist()
    
    # Obtener el ranking ordenado por puntuación
    df_ranking = df_municipios_farmacias.sort_values('PuntuaciónExtendida', ascending=False).reset_index(drop=True)
    
    # Mostrar información sobre la selección por defecto
    if len(df_ranking) >= 2:
            st.info(f"💡 **Selección automática**: Por defecto se comparan el **#{1} {df_ranking.iloc[0]['Nombre_Mostrar']}** (puntuación: {df_ranking.iloc[0]['PuntuaciónExtendida']:.2f}) y el **#{2} {df_ranking.iloc[1]['Nombre_Mostrar']}** (puntuación: {df_ranking.iloc[1]['PuntuaciónExtendida']:.2f}) del ranking.")
    
    # Selectores de municipios con valores por defecto del ranking
    col1, col2 = st.columns(2)
    
    with col1:
            # Por defecto seleccionar el primer municipio del ranking
            municipio1_default = df_ranking.iloc[0]['Nombre_Mostrar'] if len(df_ranking) > 0 else municipios_disponibles[0] if municipios_disponibles else None
            municipio1 = st.selectbox(
                "Selecciona el primer municipio:",
                options=municipios_disponibles,
                index=municipios_disponibles.index(municipio1_default) if municipio1_default in municipios_disponibles else 0,
                key="municipio1_selector"
            )
    
    with col2:
            # Por defecto seleccionar el segundo municipio del ranking
            municipio2_default = df_ranking.iloc[1]['Nombre_Mostrar'] if len(df_ranking) > 1 else None
            municipios_disponibles_2 = [m for m in municipios_disponibles if m != municipio1]
            municipio2_index = 0
            if municipio2_default and municipio2_default in municipios_disponibles_2:
                municipio2_index = municipios_disponibles_2.index(municipio2_default)
            
            municipio2 = st.selectbox(
                "Selecciona el segundo municipio:",
                options=municipios_disponibles_2,
                index=municipio2_index,
                key="municipio2_selector"
            )
    
    if municipio1 and municipio2:
            # Obtener datos de ambos municipios
            datos1 = df_municipios_farmacias[df_municipios_farmacias['Nombre_Mostrar'] == municipio1].iloc[0]
            datos2 = df_municipios_farmacias[df_municipios_farmacias['Nombre_Mostrar'] == municipio2].iloc[0]
            
            # Crear comparación visual
            st.subheader(f"🔍 Comparación: {municipio1} vs {municipio2}")
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"Puntuación Total - {municipio1}",
                    value=f"{datos1['PuntuaciónExtendida']:.2f}",
                    delta=f"{datos1['PuntuaciónExtendida'] - datos2['PuntuaciónExtendida']:.2f}" if datos1['PuntuaciónExtendida'] != datos2['PuntuaciónExtendida'] else "0.00"
                )
            
            with col2:
                st.metric(
                    label=f"Puntuación Total - {municipio2}",
                    value=f"{datos2['PuntuaciónExtendida']:.2f}",
                    delta=f"{datos2['PuntuaciónExtendida'] - datos1['PuntuaciónExtendida']:.2f}" if datos2['PuntuaciónExtendida'] != datos1['PuntuaciónExtendida'] else "0.00"
                )
            
            with col3:
                diferencia = abs(datos1['PuntuaciónExtendida'] - datos2['PuntuaciónExtendida'])
                porcentaje_dif = (diferencia / max(datos1['PuntuaciónExtendida'], datos2['PuntuaciónExtendida'])) * 100
                st.metric(
                    label="Diferencia",
                    value=f"{diferencia:.2f}",
                    delta=f"{porcentaje_dif:.1f}%"
                )
            
            # Gráfico de comparación de puntuaciones
            st.subheader("📈 Comparación de Puntuaciones")
            
            # Datos para el gráfico
            comparacion_data = {
                'Municipio': [municipio1, municipio2],
                'Puntuación Base': [datos1['Puntuación'], datos2['Puntuación']],
                'Factor': [datos1['Factor'], datos2['Factor']],
                'Puntuación con Factor': [datos1['PuntuaciónFinal'], datos2['PuntuaciónFinal']],
                'Suma Municipios Cercanos': [datos1['SumaMunicipiosCercanos'], datos2['SumaMunicipiosCercanos']],
                'Puntuación Total': [datos1['PuntuaciónExtendida'], datos2['PuntuaciónExtendida']]
            }
            
            df_comparacion = pd.DataFrame(comparacion_data)
            
            # Gráfico de barras comparativo
            fig_comparacion = px.bar(
                df_comparacion,
                x='Municipio',
                y=['Puntuación Base', 'Puntuación con Factor', 'Suma Municipios Cercanos', 'Puntuación Total'],
                title="Desglose de Puntuaciones por Componente",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
            fig_comparacion.update_layout(
                yaxis_title="Puntuación",
                xaxis_title="Municipio",
                height=500
            )
            st.plotly_chart(fig_comparacion, use_container_width=True)
            
            # Tabla comparativa detallada
            st.subheader("📋 Tabla Comparativa Detallada")
            
            tabla_comparacion = pd.DataFrame({
                'Métrica': [
                    'Puntuación Base',
                    'Factor Aplicado',
                    'Puntuación con Factor',
                    'Suma Municipios Cercanos (≤{} km)'.format(radio_km),
                    'Puntuación Total Final',
                    'Latitud',
                    'Longitud'
                ],
                municipio1: [
                    f"{datos1['Puntuación']:.2f}",
                    f"{datos1['Factor']:.4f}",
                    f"{datos1['PuntuaciónFinal']:.2f}",
                    f"{datos1['SumaMunicipiosCercanos']:.2f}",
                    f"{datos1['PuntuaciónExtendida']:.2f}",
                    f"{datos1['Latitud']:.6f}" if pd.notna(datos1['Latitud']) else "N/A",
                    f"{datos1['Longitud']:.6f}" if pd.notna(datos1['Longitud']) else "N/A"
                ],
                municipio2: [
                    f"{datos2['Puntuación']:.2f}",
                    f"{datos2['Factor']:.4f}",
                    f"{datos2['PuntuaciónFinal']:.2f}",
                    f"{datos2['SumaMunicipiosCercanos']:.2f}",
                    f"{datos2['PuntuaciónExtendida']:.2f}",
                    f"{datos2['Latitud']:.6f}" if pd.notna(datos2['Latitud']) else "N/A",
                    f"{datos2['Longitud']:.6f}" if pd.notna(datos2['Longitud']) else "N/A"
                ]
            })
            
            st.dataframe(tabla_comparacion, use_container_width=True)
            
            # Desglose detallado por indicadores
            st.subheader("🔍 Desglose Detallado por Indicadores")
            
            # Obtener datos originales para ambos municipios
            territorio1_original = datos1['Territorio']
            territorio2_original = datos2['Territorio']
            
            df_territorio1 = df_original[df_original["Territorio"] == territorio1_original]
            df_territorio2 = df_original[df_original["Territorio"] == territorio2_original]
            
            if not df_territorio1.empty and not df_territorio2.empty:
                # Crear desglose comparativo
                desglose_comparativo = []
                
                # Obtener todos los indicadores únicos
                indicadores_unicos = set(df_territorio1['Medida'].unique()) | set(df_territorio2['Medida'].unique())
                
                for indicador in sorted(indicadores_unicos):
                    clave_norm = normaliza_nombre_indicador(indicador)
                    peso = pesos.get(clave_norm, 1.0)
                    
                    # Buscar valor para municipio 1
                    valor1 = df_territorio1[df_territorio1['Medida'] == indicador]['Valor'].iloc[0] if not df_territorio1[df_territorio1['Medida'] == indicador].empty else None
                    contribucion1 = valor1 * peso if pd.notna(valor1) else 0
                    
                    # Buscar valor para municipio 2
                    valor2 = df_territorio2[df_territorio2['Medida'] == indicador]['Valor'].iloc[0] if not df_territorio2[df_territorio2['Medida'] == indicador].empty else None
                    contribucion2 = valor2 * peso if pd.notna(valor2) else 0
                    
                    desglose_comparativo.append({
                        'Indicador': indicador,
                        'Peso': f"{peso:.2f}",
                        f'Valor - {municipio1}': f"{valor1:.2f}" if pd.notna(valor1) else "N/A",
                        f'Contribución - {municipio1}': f"{contribucion1:.2f}",
                        f'Valor - {municipio2}': f"{valor2:.2f}" if pd.notna(valor2) else "N/A",
                        f'Contribución - {municipio2}': f"{contribucion2:.2f}",
                        'Diferencia': f"{contribucion1 - contribucion2:.2f}"
                    })
                
                df_desglose_comparativo = pd.DataFrame(desglose_comparativo)
                st.dataframe(df_desglose_comparativo, use_container_width=True, height=600)
                
                # Gráfico de contribuciones por indicador
                st.subheader("📊 Contribuciones por Indicador")
                
                # Preparar datos para el gráfico
                indicadores_grafico = []
                contribuciones1 = []
                contribuciones2 = []
                
                for _, row in df_desglose_comparativo.iterrows():
                    if row['Indicador'] and row[f'Contribución - {municipio1}'] != "0.00" and row[f'Contribución - {municipio2}'] != "0.00":
                        indicadores_grafico.append(row['Indicador'][:30] + "..." if len(row['Indicador']) > 30 else row['Indicador'])
                        contribuciones1.append(float(row[f'Contribución - {municipio1}']))
                        contribuciones2.append(float(row[f'Contribución - {municipio2}']))
                
                if indicadores_grafico:
                    fig_contribuciones = px.bar(
                        x=indicadores_grafico,
                        y=[contribuciones1, contribuciones2],
                        title="Comparación de Contribuciones por Indicador",
                        labels={'x': 'Indicador', 'y': 'Contribución a la Puntuación'},
                        barmode='group'
                    )
                    fig_contribuciones.update_layout(
                        xaxis_tickangle=-45,
                        height=500
                    )
                    st.plotly_chart(fig_contribuciones, use_container_width=True)
            
            # Mapa de comparación
            st.subheader("🗺️ Ubicación de los Municipios")
            
            # Crear mapa centrado entre ambos municipios
            if pd.notna(datos1['Latitud']) and pd.notna(datos1['Longitud']) and pd.notna(datos2['Latitud']) and pd.notna(datos2['Longitud']):
                lat_centro = (datos1['Latitud'] + datos2['Latitud']) / 2
                lon_centro = (datos1['Longitud'] + datos2['Longitud']) / 2
                
                m_comparacion = folium.Map(location=[lat_centro, lon_centro], zoom_start=10)
                
                # Agregar marcadores para ambos municipios
                folium.Marker(
                    [datos1['Latitud'], datos1['Longitud']],
                    popup=f"<b>{municipio1}</b><br>Puntuación: {datos1['PuntuaciónExtendida']:.2f}",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m_comparacion)
                
                folium.Marker(
                    [datos2['Latitud'], datos2['Longitud']],
                    popup=f"<b>{municipio2}</b><br>Puntuación: {datos2['PuntuaciónExtendida']:.2f}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m_comparacion)
                
                # Agregar línea conectora
                folium.PolyLine(
                    locations=[[datos1['Latitud'], datos1['Longitud']], [datos2['Latitud'], datos2['Longitud']]],
                    color='green',
                    weight=2,
                    opacity=0.7
                ).add_to(m_comparacion)
                
                st_folium(m_comparacion, width=1200, height=400)
            else:
                st.warning("No se pueden mostrar las coordenadas en el mapa para uno o ambos municipios.")
            
            # Descarga de datos de comparación
            st.subheader("📥 Descargar Datos de Comparación")
            
            csv_buffer_comparacion = BytesIO()
            df_desglose_comparativo.to_csv(csv_buffer_comparacion, index=False, sep=';', encoding='utf-8')
            csv_buffer_comparacion.seek(0)
            
            st.download_button(
                label="📥 Descargar desglose comparativo en CSV",
                data=csv_buffer_comparacion,
                file_name=f"comparacion_{municipio1}_vs_{municipio2}.csv",
                mime="text/csv"
            )

# --------------------
# Version information in the sidebar
st.sidebar.subheader("Version 1.8.0")


