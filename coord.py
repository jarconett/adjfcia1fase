import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import unicodedata
import re
from math import radians, cos, sin, asin, sqrt
from io import BytesIO
import numpy as np
from types import SimpleNamespace
import os
import requests
import json

# Set the title of the Streamlit application
st.title("Mapa Interactivo de las Farmacias de la Primera fase de Adjudicaciones de Andalucía")

# Importar NUEVO motor de proyecciones (entidades singulares)
try:
    from proyeccion_entidades_singulares_final import render_proyeccion_entidades_singulares
    motor_entidades_disponible = True
except Exception as e:
    motor_entidades_disponible = False
    st.sidebar.error(f"Error importando motor de proyecciones nuevo: {e}")

# --------------------
# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Mapa y Ranking", "📊 Comparación de Municipios", "📈 Proyecciones Demográficas (Entidades singulares)"])

# --------------------
# Configuración de Normalización (FUERA de los tabs)
st.sidebar.header("🔧 Configuración de Normalización")

# Opciones de normalización
metodo_normalizacion = st.sidebar.selectbox(
    "Método de normalización:",
    ["Min-Max (0-1)", "Min-Max (0-100)", "Min-Max Logarítmico (0-1)", "Min-Max Logarítmico (0-100)", "Z-Score", "Sin normalizar"],
    index=3
)

# Escala de normalización
if "Min-Max" in metodo_normalizacion:
    escala_max = 1.0 if "0-1" in metodo_normalizacion else 100.0
else:
    escala_max = 1.0

# Configuración de rango personalizado para normalización
if "Min-Max" in metodo_normalizacion:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Rango de Normalización")

    usar_rango_personalizado = st.sidebar.checkbox(
        "Usar rango personalizado",
        value=False,
        help="Permite establecer un valor máximo personalizado para la normalización"
    )

    if usar_rango_personalizado:
        valor_max_personalizado = st.sidebar.number_input(
            "Valor máximo para normalización:",
            min_value=0.0,
            value=100.0,
            step=1.0,
            help="Valor que se usará como máximo (100) en la normalización"
        )
    else:
        valor_max_personalizado = None
else:
    valor_max_personalizado = None

# Configuración de aplicación del Factor
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Aplicación del Factor")

aplicar_factor_antes = st.sidebar.checkbox(
    "Aplicar Factor antes de normalización",
    value=True,
    help="Si está activado, el Factor se aplica a cada indicador antes de normalizar. Si no, se aplica a la puntuación final."
)

if aplicar_factor_antes:
    st.sidebar.info("🔧 **Factor aplicado a indicadores individuales** antes de normalización")
else:
    st.sidebar.info("🔧 **Factor aplicado a puntuación final** (comportamiento actual)")

# Información sobre direccionalidad
st.sidebar.info("💡 **Direccionalidad**: Se controla con los pesos positivos/negativos en los sliders")

# --------------------
# Sidebar for user inputs
st.sidebar.header("Configuración de Datos y Puntuación")

# --------------------
# TAB 1: Mapa y Ranking
with tab1:
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
        "Consultorio.csv",
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
            # Leer el archivo especificando explícitamente los nombres de columnas
            df_farmacias = pd.read_csv(
                territorios_file.name, 
                sep=";", 
                na_values=["-", "", "NA"],
                names=['Territorio', 'Latitud', 'Longitud', 'Factor', 'Singular', 'Provincia', 'Ldo']
            )
            # Saltar la primera fila que contiene los encabezados
            df_farmacias = df_farmacias.iloc[1:].reset_index(drop=True)
            df_farmacias.columns = df_farmacias.columns.str.strip()

            # Convertir las columnas numéricas al tipo correcto
            df_farmacias['Latitud'] = pd.to_numeric(df_farmacias['Latitud'], errors='coerce')
            df_farmacias['Longitud'] = pd.to_numeric(df_farmacias['Longitud'], errors='coerce')
            df_farmacias['Factor'] = pd.to_numeric(df_farmacias['Factor'], errors='coerce')
            # Rellenar valores faltantes en Factor con 1.0
            df_farmacias['Factor'] = df_farmacias['Factor'].fillna(1.0)

            # Verificación de datos cargados correctamente

            # Información de carga exitosa
            st.sidebar.success(f"✅ Archivo Territorios.csv cargado correctamente")

            if 'Singular' in df_farmacias.columns:
                # Crear Nombre_Mostrar único combinando Territorio + Singular
                df_farmacias['Nombre_Mostrar'] = df_farmacias.apply(
                    lambda row: f"{row['Territorio']} ({row['Singular'].strip()})" if pd.notna(row['Singular']) and str(row['Singular']).strip() != ''
                    else f"{row['Territorio']}", axis=1
                )
            else:
                df_farmacias['Nombre_Mostrar'] = df_farmacias['Territorio']
            # 🔧 Asegurar unicidad absoluta de Nombre_Mostrar (por si hay duplicados exactos)
            df_farmacias['Nombre_Mostrar'] = df_farmacias['Nombre_Mostrar'].astype(str)
            df_farmacias['Nombre_Mostrar'] = df_farmacias['Nombre_Mostrar'] + \
                df_farmacias.groupby('Nombre_Mostrar').cumcount().replace(0, '').astype(str)

            st.sidebar.success(f"Farmacias cargadas: {len(df_farmacias)} registros")

            # --- Integrar coordenadas desde Territorios.csv con prioridad ---
            try:
                coords_territorios = df_farmacias[["Territorio", "Latitud", "Longitud"]].copy()
                # Usar solo filas con coordenadas válidas y una por territorio
                coords_territorios = coords_territorios.dropna(subset=["Latitud", "Longitud"])\
                                                 .drop_duplicates(subset=["Territorio"], keep="first")

                if not coords_territorios.empty:
                    if df_coords_existentes.empty:
                        # Si no hay coordenadas IECA, usar las de Territorios.csv directamente
                        df_coords_existentes = coords_territorios.copy()
                    else:
                        # Actualizar coordenadas existentes y añadir las que falten
                        base_idx = df_coords_existentes.set_index("Territorio")
                        terr_idx = coords_territorios.set_index("Territorio")

                        # Sobrescribir Latitud/Longitud donde Territorios.csv tenga valores
                        base_idx.update(terr_idx[["Latitud", "Longitud"]])

                        # Añadir territorios nuevos que solo están en Territorios.csv
                        missing = terr_idx.index.difference(base_idx.index)
                        if len(missing) > 0:
                            base_idx = pd.concat([base_idx, terr_idx.loc[missing]], axis=0)

                        df_coords_existentes = base_idx.reset_index()

                    st.sidebar.info("Coordenadas de Territorios.csv priorizadas sobre IECA para territorios incluidos.")
            except Exception as e:
                st.sidebar.warning(f"No se pudieron integrar coordenadas de Territorios.csv: {e}")

        except Exception as e:
            st.sidebar.error(f"Error al leer Territorios.csv: {e}")
    else:
        st.sidebar.error("No se encontró el archivo Territorios.csv")

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
        """Construye el nombre del indicador combinando Medida y columnas extra.
        Regla: si 'Medida' es genérica (p.ej. 'Centros', 'Nº de oficinas', 'Espacios deportivos')
        o si en el archivo solo hay una 'Medida', priorizar valores de columnas extra como base del nombre.
        Además, evitar incluir 'Singular' en Consultorio.
        """
        # Definir medidas genéricas conocidas (normalizadas)
        medidas_genericas = {
            'centros', 'n_de_oficinas', 'numero_de_oficinas', 'número_de_oficinas',
            'numero_de_centros', 'número_de_centros', 'espacios_deportivos', 'establecimientos',
        }

        medida_raw = str(row.get('Medida', '')).strip()
        medida_norm = normaliza_nombre_indicador(medida_raw)

        # Preparar lista de columnas extra priorizadas
        prioridad_extras = [
            'Tipo de centro', 'Tipo de instalación', 'Nivel educativo',
            'Titularidad', 'Tipo', 'Categoría', 'Sexo', 'Edad'
        ]
        # Ordenar las extras colocando primero las que estén en prioridad_extras
        extras_ordenadas = sorted(
            extras,
            key=lambda c: (0 if c in prioridad_extras else 1, prioridad_extras.index(c) if c in prioridad_extras else 999, c)
        )

        # Consultorio: no incluir 'Singular' en el nombre
        omitir_singular = ('Singular' in extras) and bool(str(row.get('Singular', '')).strip())

        # Decidir si priorizamos extras sobre medida
        priorizar_extras = (medida_norm in medidas_genericas)

        parts = []

        if not priorizar_extras:
            if medida_raw:
                parts.append(medida_raw)

        for col in extras_ordenadas:
            if omitir_singular and col == 'Singular':
                continue
            val = str(row.get(col, '')).strip()
            if val and val.lower() not in ['nan', 'none', 'na', '']:
                parts.append(val)

        # Si no agregamos nada (p.ej., valores vacíos), volver a medida
        if not parts and medida_raw:
            parts = [medida_raw]

        clean_parts = [limpiar_texto(p) for p in parts]
        return "_".join(clean_parts)

    def obtener_indicadores_unicos_por_archivo(df_archivo: pd.DataFrame) -> list:
        """Genera la lista de indicadores únicos (human-readable) para un archivo.
        Usa combinación de Medida + columnas extra relevantes, deduplicando por todas ellas.
        """
        if df_archivo is None or df_archivo.empty:
            return []

        columnas_basicas = {'Territorio', 'Medida', 'Valor', '__archivo__'}
        columnas_extra = [col for col in df_archivo.columns if col not in columnas_basicas]

        # Trabajar solo con columnas relevantes y deduplicar combinaciones
        cols_para_unicos = ['Medida'] + columnas_extra
        df_tmp = df_archivo[cols_para_unicos].copy()

        # Normalizar valores a texto para deduplicación estable
        for col in cols_para_unicos:
            df_tmp[col] = df_tmp[col].astype(str).str.strip()

        df_tmp = df_tmp.drop_duplicates()

        # Construir etiqueta usando la misma función que usa la UI
        indicadores = df_tmp.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1).unique().tolist()
        return sorted(indicadores)

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
        # Normalizar unicode y quitar acentos
        nombre = unicodedata.normalize('NFKD', nombre)
        nombre = ''.join([c for c in nombre if not unicodedata.combining(c)])
        # Convertir a minúsculas
        nombre = nombre.lower()
        # Quitar solo caracteres especiales problemáticos, mantener letras, números, espacios y guiones
        nombre = re.sub(r'[^\w\s\-]', '', nombre)
        # Limpiar espacios múltiples
        nombre = re.sub(r'\s+', ' ', nombre)
        return nombre.strip()

    def normalizar_indicador(valor, min_val, max_val, direccion='alto_deseable'):
        """
        Normaliza un valor a escala 0-1
        direccion: 'alto_deseable' o 'bajo_deseable'
        """
        if pd.isna(valor) or pd.isna(min_val) or pd.isna(max_val):
            return 0.0

        # Evitar división por cero
        if max_val == min_val:
            return 0.5

        # Normalizar a escala 0-1
        normalizado = (valor - min_val) / (max_val - min_val)

        # Aplicar direccionalidad
        if direccion == 'bajo_deseable':
            normalizado = 1.0 - normalizado

        # Asegurar que esté en rango [0, 1]
        return max(0.0, min(1.0, normalizado))

    def normalizar_indicador_logaritmico(valor, min_val, max_val, direccion='alto_deseable'):
        """
        Normaliza un valor usando escala logarítmica (útil para indicadores con gran disparidad)
        direccion: 'alto_deseable' o 'bajo_deseable'
        """
        if pd.isna(valor) or pd.isna(min_val) or pd.isna(max_val):
            return 0.0

        # Evitar valores negativos o cero para logaritmo
        valor_ajustado = max(valor, 1.0)
        min_ajustado = max(min_val, 1.0)
        max_ajustado = max(max_val, 1.0)

        # Evitar división por cero
        if max_ajustado == min_ajustado:
            return 0.5

        # Aplicar logaritmo
        import numpy as np
        log_valor = np.log(valor_ajustado)
        log_min = np.log(min_ajustado)
        log_max = np.log(max_ajustado)

        # Normalizar en escala logarítmica
        normalizado = (log_valor - log_min) / (log_max - log_min)

        # Aplicar direccionalidad
        if direccion == 'bajo_deseable':
            normalizado = 1.0 - normalizado

        # Asegurar que esté en rango [0, 1]
        return max(0.0, min(1.0, normalizado))

    def calcular_estadisticas_indicador(serie):
        """Calcula estadísticas para normalización de un indicador"""
        serie_limpia = serie.dropna()
        if len(serie_limpia) == 0:
            return {'min': 0, 'max': 1, 'mean': 0.5, 'std': 0}

        return {
            'min': serie_limpia.min(),
            'max': serie_limpia.max(),
            'mean': serie_limpia.mean(),
            'std': serie_limpia.std()
        }

    def obtener_poblacion_territorio(territorio, singular=None):
        """Obtiene la población de un territorio desde singular_pob_sexo.csv"""
        try:
            # Cargar el archivo singular_pob_sexo.csv
            df_singular_pob = pd.read_csv("singular_pob_sexo.csv", sep=";", na_values=["-", "", "NA"])

            # Determinar el nombre a buscar
            nombre_a_buscar = None
            if singular and pd.notna(singular) and str(singular).strip() != '':
                # Si Singular tiene valor, usar ese
                nombre_a_buscar = str(singular).strip()
            else:
                # Si Singular está vacío, usar Territorio
                nombre_a_buscar = str(territorio).strip()

            # Buscar población para "Ambos sexos"
            if nombre_a_buscar:
                poblacion_data = df_singular_pob[
                    (df_singular_pob['Territorio'] == nombre_a_buscar) & 
                    (df_singular_pob['Sexo'] == 'Ambos sexos') &
                    (df_singular_pob['Medida'] == 'Población')
                ]

                if not poblacion_data.empty:
                    valor = poblacion_data.iloc[0]['Valor']
                    return f"{valor:.0f}" if pd.notna(valor) else "N/A"

            return "N/A"
        except Exception as e:
            return "N/A"

    def obtener_poblacion_territorio_con_factor(territorio, singular=None, factor=None):
        """Obtiene población para el Territorio desde singular_pob_sexo.csv.
        Regla: si la fila tiene 'Singular' (no vacío), usar SIEMPRE el Territorio para buscar y multiplicar por Factor.
        Si no hay 'Singular', devolver solo la población del Territorio sin factor.
        """
        try:
            df_singular_pob = pd.read_csv("singular_pob_sexo.csv", sep=";", na_values=["-", "", "NA"])
            nombre_a_buscar = str(territorio).strip()
            # Coincidencia robusta: normalizar ambos lados para manejar acentos/guiones
            df_singular_pob = df_singular_pob.copy()
            df_singular_pob['__terr_norm'] = df_singular_pob['Territorio'].astype(str).apply(normalizar_nombre_municipio)
            target_norm = normalizar_nombre_municipio(nombre_a_buscar)

            # Intentar orden estándar (Sexo='Ambos sexos', Medida='Población')
            match_std = df_singular_pob[
                (df_singular_pob['__terr_norm'] == target_norm) &
                (df_singular_pob.get('Sexo', '') == 'Ambos sexos') &
                (df_singular_pob.get('Medida', '') == 'Población')
            ]
            # Intentar orden alternativo (Medida='Ambos sexos', Sexo='Población')
            match_alt = df_singular_pob[
                (df_singular_pob['__terr_norm'] == target_norm) &
                (df_singular_pob.get('Medida', '') == 'Ambos sexos') &
                (df_singular_pob.get('Sexo', '') == 'Población')
            ]
            poblacion_data = match_std if not match_std.empty else match_alt
            if poblacion_data.empty:
                return "N/A"

            valor = poblacion_data.iloc[0]['Valor']
            if pd.isna(valor):
                return "N/A"

            tiene_singular = bool(singular) and pd.notna(singular) and str(singular).strip() != ''
            if tiene_singular and factor is not None and pd.notna(factor):
                try:
                    return f"{(float(valor) * float(factor)):.0f}"
                except Exception:
                    return f"{float(valor):.0f}"
            else:
                return f"{float(valor):.0f}"
        except Exception:
            return "N/A"

    def calcular_ruta_osrm(lat_origen, lon_origen, lat_destino, lon_destino):
        """Calcula distancia y tiempo de ruta usando OSRM (Open Source Routing Machine).
        Retorna: (distancia_km, tiempo_minutos, ruta_coordenadas) o None si falla.
        """
        try:
            # Usar el servicio público de OSRM para routing
            url = f"http://router.project-osrm.org/route/v1/driving/{lon_origen},{lat_origen};{lon_destino},{lat_destino}?overview=full&geometries=geojson"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and len(data.get('routes', [])) > 0:
                    route = data['routes'][0]
                    distancia_metros = route['distance']
                    tiempo_segundos = route['duration']
                    
                    distancia_km = distancia_metros / 1000.0
                    tiempo_minutos = tiempo_segundos / 60.0
                    
                    # Extraer coordenadas de la ruta
                    ruta_coordenadas = route['geometry']['coordinates']
                    # Convertir de [lon, lat] a [lat, lon] para folium
                    ruta_coordenadas = [[coord[1], coord[0]] for coord in ruta_coordenadas]
                    
                    return distancia_km, tiempo_minutos, ruta_coordenadas
        except Exception:
            pass
        
        return None

    def calcular_ruta_estimada(lat_origen, lon_origen, lat_destino, lon_destino):
        """Calcula distancia y tiempo estimado basado en distancia geodésica.
        Usa un factor de corrección para rutas por carretera (1.4x) y velocidad promedio de 80 km/h.
        Retorna: (distancia_km, tiempo_minutos, None)
        """
        try:
            # Calcular distancia geodésica en línea recta
            distancia_recta_km = geodesic((lat_origen, lon_origen), (lat_destino, lon_destino)).kilometers
            
            # Factor de corrección para rutas por carretera (típicamente 1.3-1.5x más largo)
            factor_carretera = 1.4
            distancia_carretera_km = distancia_recta_km * factor_carretera
            
            # Velocidad promedio en carretera (km/h)
            velocidad_promedio = 80.0
            
            # Calcular tiempo en minutos
            tiempo_horas = distancia_carretera_km / velocidad_promedio
            tiempo_minutos = tiempo_horas * 60.0
            
            return distancia_carretera_km, tiempo_minutos, None
        except Exception:
            return None

    def obtener_coordenadas_destino(destino_texto):
        """Obtiene coordenadas de una localidad usando geocodificación.
        Primero intenta con coordenadas conocidas, luego con geocodificación.
        """
        # Diccionario de coordenadas conocidas para ciudades comunes en Andalucía
        coordenadas_conocidas = {
            'granada': (37.1773, -3.5986),
            'granada, españa': (37.1773, -3.5986),
            'granada españa': (37.1773, -3.5986),
            'sevilla': (37.3891, -5.9845),
            'sevilla, españa': (37.3891, -5.9845),
            'sevilla españa': (37.3891, -5.9845),
            'málaga': (36.7213, -4.4214),
            'málaga, españa': (36.7213, -4.4214),
            'malaga': (36.7213, -4.4214),
            'malaga, españa': (36.7213, -4.4214),
            'córdoba': (37.8882, -4.7794),
            'córdoba, españa': (37.8882, -4.7794),
            'cordoba': (37.8882, -4.7794),
            'cordoba, españa': (37.8882, -4.7794),
            'cádiz': (36.5270, -6.2886),
            'cádiz, españa': (36.5270, -6.2886),
            'cadiz': (36.5270, -6.2886),
            'cadiz, españa': (36.5270, -6.2886),
            'jaén': (37.7796, -3.7849),
            'jaén, españa': (37.7796, -3.7849),
            'jaen': (37.7796, -3.7849),
            'jaen, españa': (37.7796, -3.7849),
            'almería': (36.8381, -2.4597),
            'almería, españa': (36.8381, -2.4597),
            'almeria': (36.8381, -2.4597),
            'almeria, españa': (36.8381, -2.4597),
            'huelva': (37.2574, -6.9499),
            'huelva, españa': (37.2574, -6.9499),
            'marbella': (36.5102, -4.8860),
            'marbella, españa': (36.5102, -4.8860),
            'algeciras': (36.1276, -5.4509),
            'algeciras, españa': (36.1276, -5.4509),
            'ronda': (36.7423, -5.1667),
            'ronda, españa': (36.7423, -5.1667),
            'ronda españa': (36.7423, -5.1667),
        }
        
        # Normalizar el texto de búsqueda
        destino_normalizado = destino_texto.lower().strip()
        
        # Primero intentar con coordenadas conocidas
        if destino_normalizado in coordenadas_conocidas:
            return coordenadas_conocidas[destino_normalizado]
        
        # Si no está en las conocidas, intentar geocodificación
        geolocator = None
        try:
            geolocator = Nominatim(user_agent="andalucia-mapa-rutas-v2")
            # Aumentar timeout y reintentos
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=3)
            
            # Lista de variaciones a intentar
            variaciones = []
            
            # Agregar el texto original
            variaciones.append(destino_texto)
            
            # Si no tiene "España", agregar variaciones con país
            if 'españa' not in destino_normalizado and 'spain' not in destino_normalizado:
                variaciones.append(f"{destino_texto}, España")
                variaciones.append(f"{destino_texto}, Andalucía, España")
                variaciones.append(f"{destino_texto}, Spain")
            
            # Intentar cada variación
            for variacion in variaciones:
                try:
                    location = geocode(variacion, timeout=20, language='es')
                    if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                        if location.latitude and location.longitude:
                            return location.latitude, location.longitude
                except Exception as e:
                    # Continuar con la siguiente variación si esta falla
                    continue
            
        except Exception as e:
            # Si hay un error general con el geolocator, continuar con búsqueda parcial
            pass
        finally:
            # Limpiar recursos si es necesario
            pass
        
        # Si todas las variaciones fallan, intentar búsqueda parcial en coordenadas conocidas
        try:
            for ciudad, coords in coordenadas_conocidas.items():
                ciudad_limpia = ciudad.replace(',', '').replace(' ', '').replace('españa', '').replace('spain', '').strip()
                destino_limpio = destino_normalizado.replace(',', '').replace(' ', '').replace('españa', '').replace('spain', '').strip()
                
                # Buscar coincidencias parciales (al menos 4 caracteres en común)
                if len(ciudad_limpia) >= 4 and len(destino_limpio) >= 4:
                    if ciudad_limpia in destino_limpio or destino_limpio in ciudad_limpia:
                        return coords
            
            # Último intento: búsqueda parcial más flexible por palabras
            for ciudad, coords in coordenadas_conocidas.items():
                ciudad_palabras = set(ciudad.lower().replace(',', '').split())
                destino_palabras = set(destino_normalizado.replace(',', '').split())
                
                # Si hay al menos una palabra en común (mínimo 3 caracteres)
                palabras_comunes = [p for p in ciudad_palabras if len(p) >= 3 and p in destino_palabras]
                if palabras_comunes:
                    return coords
        except Exception:
            pass
        
        return None, None

    # Configuración de normalización ya definida fuera de los tabs

    # --------------------
    # Load Weights from CSV
    st.sidebar.subheader("Cargar/Guardar Pesos")
    uploaded_weights_file = st.sidebar.file_uploader(
        "Sube un archivo CSV con pesos guardados", type="csv", key="weights_uploader"
    )
    loaded_pesos_dict = {}
    # Cargar automáticamente desde pesos_guardados.csv en la raíz si existe
    try:
        if os.path.exists("pesos_guardados.csv"):
            df_loaded_pesos = pd.read_csv("pesos_guardados.csv", sep=';')
            if 'Indicador' in df_loaded_pesos.columns and 'Peso' in df_loaded_pesos.columns:
                loaded_pesos_dict = pd.Series(df_loaded_pesos.Peso.values, index=df_loaded_pesos.Indicador).to_dict()
                st.sidebar.success("Pesos cargados automáticamente desde pesos_guardados.csv")
            else:
                st.sidebar.warning("pesos_guardados.csv no contiene las columnas 'Indicador' y 'Peso'.")
    except Exception as e:
        st.sidebar.error(f"Error cargando pesos_guardados.csv: {e}")
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
    st.subheader("Ajuste de Pesos y Parámetros")

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
            indicadores_combinados = obtener_indicadores_unicos_por_archivo(df_archivo)

            # Campo para valor global y botón fuera del form (permitido)
            col1, col2 = st.columns([0.7, 0.3])

            # Determinar valor global por defecto según reglas específicas
            valor_global_por_defecto = 1.0  # Valor por defecto general

            # Regla 1: Todos los indicadores de explot_ganaderas tienen peso 0 por defecto
            if "explot_ganaderas" in archivo:
                valor_global_por_defecto = 0.0

            valor_global = col1.number_input(
                f"Valor global para {archivo}", -5.0, 5.0, valor_global_por_defecto, 0.1, key=f"global_val_{archivo}"
            )
            if col2.button("Aplicar", key=f"aplicar_{archivo}"):
                for indicador_completo in sorted(indicadores_combinados):
                    clave_norm = normaliza_nombre_indicador(indicador_completo)
                    st.session_state[f"{archivo}-{clave_norm}"] = valor_global
                st.rerun()

            # Mostrar indicadores detectados para depuración opcional
            with st.expander("Indicadores detectados", expanded=False):
                st.caption(f"Se han detectado {len(indicadores_combinados)} indicadores únicos")
                st.write(indicadores_combinados)

            # Sliders individuales
            for indicador_completo in sorted(indicadores_combinados):
                clave_norm = normaliza_nombre_indicador(indicador_completo)

                # Determinar peso por defecto según reglas específicas
                peso_por_defecto = 1.0  # Valor por defecto general

                # Regla 1: Todos los indicadores de explot_ganaderas tienen peso 0 por defecto
                if "explot_ganaderas" in archivo:
                    peso_por_defecto = 0.0

                # Regla 2: Indicadores que empiecen por "Superficie" en centros_asistenciales tienen peso 0 por defecto
                if "centros_asistenciales" in archivo and indicador_completo.startswith("Superficie"):
                    peso_por_defecto = 0.0

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

    columnas_basicas = {'Territorio', 'Medida', 'Valor'}
    columnas_extra = [col for col in df_original.columns if col not in columnas_basicas and col != '__archivo__']
    df_original['Medida'] = df_original.apply(lambda row: combinar_medida_y_extras(row, columnas_extra), axis=1)

@st.cache_data
def preparar_datos_base(df_original, df_coords, df_farmacias, metodo_normalizacion, escala_max, valor_max_personalizado=None, aplicar_factor_antes=False):
        df_pivot = df_original.pivot_table(
            index="Territorio", columns="Medida", values="Valor", aggfunc="first"
        ).reset_index()
        col_map = {col: normaliza_nombre_indicador(col) if col != 'Territorio' else col for col in df_pivot.columns}
        df_pivot = df_pivot.rename(columns=col_map)
        df_pivot["Territorio_normalizado"] = df_pivot["Territorio"].apply(normalizar_nombre_municipio)

        # Aplicar Factor a indicadores individuales antes de normalización si está habilitado
        if aplicar_factor_antes:
            # Obtener factores de farmacias
            factores_dict = dict(zip(df_farmacias['Territorio'], df_farmacias['Factor']))

            # Aplicar factor a cada indicador para cada territorio
            columnas_indicadores = [col for col in df_pivot.columns if col not in ['Territorio', 'Territorio_normalizado']]
            for col in columnas_indicadores:
                if col in df_pivot.columns:
                    df_pivot[col] = df_pivot.apply(
                        lambda row: row[col] * factores_dict.get(row['Territorio'], 1.0) if pd.notna(row[col]) else row[col],
                        axis=1
                    )

        # Aplicar normalización si está habilitada
        if metodo_normalizacion != "Sin normalizar":
            df_pivot_normalizado = df_pivot.copy()

            # Obtener columnas de indicadores (excluyendo Territorio y Territorio_normalizado)
            columnas_indicadores = [col for col in df_pivot.columns if col not in ['Territorio', 'Territorio_normalizado']]

            for col in columnas_indicadores:
                if col in df_pivot.columns:
                    serie_original = df_pivot[col]
                    serie_limpia = serie_original.dropna()

                    if len(serie_limpia) > 0:
                        if "Min-Max" in metodo_normalizacion:
                            min_val = serie_limpia.min()

                            # Usar valor máximo personalizado si está configurado
                            if valor_max_personalizado is not None:
                                max_val = valor_max_personalizado
                            else:
                                max_val = serie_limpia.max()

                            # Aplicar normalización Min-Max (direccionalidad se maneja con pesos)
                            if "Logarítmico" in metodo_normalizacion:
                                # Usar normalización logarítmica
                                df_pivot_normalizado[col] = serie_original.apply(
                                    lambda x: normalizar_indicador_logaritmico(x, min_val, max_val, 'alto_deseable') * escala_max
                                    if pd.notna(x) else 0
                                )
                            else:
                                # Usar normalización lineal
                                df_pivot_normalizado[col] = serie_original.apply(
                                    lambda x: normalizar_indicador(x, min_val, max_val, 'alto_deseable') * escala_max
                                    if pd.notna(x) else 0
                                )
                        elif metodo_normalizacion == "Z-Score":
                            mean_val = serie_limpia.mean()
                            std_val = serie_limpia.std()

                            if std_val > 0:
                                # Normalizar Z-Score y convertir a escala 0-1
                                z_scores = (serie_original - mean_val) / std_val
                                # Convertir Z-scores a escala 0-1 usando función sigmoide
                                df_pivot_normalizado[col] = 1 / (1 + np.exp(-z_scores)) * escala_max
                            else:
                                df_pivot_normalizado[col] = 0.5 * escala_max

            df_pivot = df_pivot_normalizado
        municipios_con_farmacia = set()
        df_farmacias_factores = pd.DataFrame()
        if not df_farmacias.empty:
            if 'Territorio' in df_farmacias.columns and 'Factor' in df_farmacias.columns:
                # Procesar datos de farmacias
                df_farmacias["Territorio_normalizado"] = df_farmacias["Territorio"].apply(normalizar_nombre_municipio)

                # Crear un identificador único para cada fila
                df_farmacias["ID_Unico"] = df_farmacias.index

                municipios_con_farmacia = set(df_farmacias["Territorio_normalizado"])
                # Incluir todas las columnas necesarias del archivo de farmacias
                columnas_farmacias = ["Territorio_normalizado", "Factor", "Nombre_Mostrar", "ID_Unico", "Territorio"]
                if 'Provincia' in df_farmacias.columns:
                    columnas_farmacias.append('Provincia')
                if 'Ldo' in df_farmacias.columns:
                    columnas_farmacias.append('Ldo')
                if 'Singular' in df_farmacias.columns:
                    columnas_farmacias.append('Singular')
                if 'Latitud' in df_farmacias.columns:
                    columnas_farmacias.append('Latitud')
                if 'Longitud' in df_farmacias.columns:
                    columnas_farmacias.append('Longitud')

                df_farmacias_factores = df_farmacias[columnas_farmacias].copy()
            else:
                st.sidebar.error(f"Faltan columnas 'Territorio' o 'Factor' en df_farmacias. Columnas disponibles: {list(df_farmacias.columns)}")
        else:
            st.sidebar.error("df_farmacias está vacío")

        df_con_farmacia_base = df_pivot[df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()
        df_sin_farmacia_base = df_pivot[~df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()

        # Procesamiento de datos completado

        # Información de estado
        if len(df_con_farmacia_base) > 0:
            st.sidebar.success(f"✅ {len(df_con_farmacia_base)} municipios con farmacia encontrados")
        elif len(municipios_con_farmacia) > 0:
            st.sidebar.warning(f"⚠️ No se encontraron coincidencias entre {len(municipios_con_farmacia)} municipios con farmacia y {len(df_pivot)} municipios en los datos")

            # Información de diagnóstico simplificada
            st.sidebar.write(f"Municipios con farmacia: {len(municipios_con_farmacia)}")
            st.sidebar.write(f"Municipios en datos: {len(df_pivot)}")
        # -----------------------------
        # NUEVO BLOQUE: generar df_con_farmacia_base con múltiples filas por Territorio
        # -----------------------------
        df_pivot_con = df_pivot[df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()
        df_pivot_sin = df_pivot[~df_pivot["Territorio_normalizado"].isin(municipios_con_farmacia)].copy()

        if not df_farmacias_factores.empty:
            # Merge inteligente entre pivot y farmacias considerando Territorio y Singular
            # Esto es necesario para manejar correctamente las entidades singulares de Consultorio.csv
            
            # Crear una clave compuesta para el merge que considere tanto Territorio como Singular
            df_pivot_con['merge_key'] = df_pivot_con['Territorio']
            df_farmacias_factores['merge_key'] = df_farmacias_factores['Territorio']
            
            # Para cada fila en df_pivot_con, buscar la correspondiente en df_farmacias_factores
            df_con_farmacia_base = pd.DataFrame()
            
            for idx, row_pivot in df_pivot_con.iterrows():
                territorio = row_pivot['Territorio']
                terr_norm = row_pivot.get('Territorio_normalizado', normalizar_nombre_municipio(territorio))
                
                # Buscar todas las farmacias que coincidan con este territorio (match por nombre normalizado)
                if 'Territorio_normalizado' in df_farmacias_factores.columns:
                    matches = df_farmacias_factores[df_farmacias_factores['Territorio_normalizado'] == terr_norm]
                else:
                    matches = df_farmacias_factores[df_farmacias_factores['Territorio'] == territorio]
                
                if not matches.empty:
                    # Si hay múltiples matches, crear una fila para cada uno
                    for _, match in matches.iterrows():
                        row_result = row_pivot.copy()
                        row_result['Factor'] = match['Factor']
                        row_result['Nombre_Mostrar'] = match['Nombre_Mostrar']
                        # Mantener el identificador único de la farmacia
                        if 'ID_Unico' in match.index:
                            row_result['ID_Unico'] = match['ID_Unico']
                        
                        # Agregar otras columnas si existen
                        if 'Provincia' in match.index:
                            row_result['Provincia'] = match['Provincia']
                        if 'Ldo' in match.index:
                            row_result['Ldo'] = match['Ldo']
                        if 'Singular' in match.index:
                            row_result['Singular'] = match['Singular']
                        # Coordenadas específicas por entidad singular (si existen en Territorios.csv)
                        if 'Latitud' in match.index:
                            row_result['Latitud'] = match['Latitud']
                        if 'Longitud' in match.index:
                            row_result['Longitud'] = match['Longitud']
                        
                        df_con_farmacia_base = pd.concat([df_con_farmacia_base, row_result.to_frame().T], ignore_index=True)
                else:
                    # Si no hay matches, usar valores por defecto
                    row_result = row_pivot.copy()
                    row_result['Factor'] = 1.0
                    row_result['Nombre_Mostrar'] = territorio
                    df_con_farmacia_base = pd.concat([df_con_farmacia_base, row_result.to_frame().T], ignore_index=True)
            
            # Asegurar que Factor siempre tenga valor
            df_con_farmacia_base["Factor"] = pd.to_numeric(df_con_farmacia_base["Factor"], errors="coerce").fillna(1.0)

            # Si por algún motivo Nombre_Mostrar está vacío, usar Territorio como respaldo
            df_con_farmacia_base["Nombre_Mostrar"] = df_con_farmacia_base["Nombre_Mostrar"].fillna(df_con_farmacia_base["Territorio"])

            # Añadir coordenadas: preferir las de Territorios.csv por entidad singular y completar con df_coords si faltan
            df_con_farmacia_base = pd.merge(
                df_con_farmacia_base, df_coords, on="Territorio", how="left", suffixes=("", "_coord")
            )
            if 'Latitud_coord' in df_con_farmacia_base.columns:
                df_con_farmacia_base['Latitud'] = df_con_farmacia_base['Latitud'].combine_first(df_con_farmacia_base['Latitud_coord'])
                df_con_farmacia_base['Longitud'] = df_con_farmacia_base['Longitud'].combine_first(df_con_farmacia_base['Longitud_coord'])
                df_con_farmacia_base = df_con_farmacia_base.drop(columns=[c for c in ['Latitud_coord','Longitud_coord'] if c in df_con_farmacia_base.columns])
        else:
            # Si no hay farmacias, mantener lógica previa
            df_con_farmacia_base = df_pivot_con.copy()
            df_con_farmacia_base["Factor"] = 1.0
            df_con_farmacia_base["Nombre_Mostrar"] = df_con_farmacia_base["Territorio"]
            df_con_farmacia_base = pd.merge(df_con_farmacia_base, df_coords, on="Territorio", how="left")
        # Municipios sin farmacia igual que antes
        df_sin_farmacia_base = pd.merge(df_pivot_sin, df_coords, on="Territorio", how="left")
        return df_con_farmacia_base, df_sin_farmacia_base

        #if not df_farmacias_factores.empty:
            # Hacer merge manual para manejar duplicados correctamente
            #df_con_farmacia_base['Factor'] = 1.0
            #df_con_farmacia_base['Nombre_Mostrar'] = df_con_farmacia_base['Territorio']

            # Para cada fila en df_con_farmacia_base, buscar la correspondiente en df_farmacias
            #for idx, row in df_con_farmacia_base.iterrows():
                #territorio_norm = row['Territorio_normalizado']
                #territorio_orig = row['Territorio']

                # Buscar en df_farmacias las filas que coincidan
                #matches = df_farmacias[df_farmacias['Territorio_normalizado'] == territorio_norm]

                #if not matches.empty:
                    # Si hay múltiples matches, usar el primero (o implementar lógica más sofisticada)
                    #match = matches.iloc[0]
                    #df_con_farmacia_base.at[idx, 'Factor'] = match['Factor']
                    #df_con_farmacia_base.at[idx, 'Nombre_Mostrar'] = match['Nombre_Mostrar']

                    # Agregar otras columnas si existen
                    #if 'Provincia' in df_farmacias.columns:
                        #df_con_farmacia_base.at[idx, 'Provincia'] = match['Provincia']
                    #if 'Ldo' in df_farmacias.columns:
                        #df_con_farmacia_base.at[idx, 'Ldo'] = match['Ldo']
        #else:
            #df_con_farmacia_base['Factor'] = 1.0
            #df_con_farmacia_base['Nombre_Mostrar'] = df_con_farmacia_base['Territorio']

        df_con_farmacia_base = pd.merge(df_con_farmacia_base, df_coords, on="Territorio", how="left")
        df_sin_farmacia_base = pd.merge(df_sin_farmacia_base, df_coords, on="Territorio", how="left")
        return df_con_farmacia_base, df_sin_farmacia_base

def calcular_puntuaciones(df_con_farmacia_base, df_sin_farmacia_base, pesos, radio_km, aplicar_factor_antes=False):
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
    # Asegurar que Factor sea numérico
    df_con_farmacia['Factor'] = pd.to_numeric(df_con_farmacia['Factor'], errors='coerce').fillna(1.0)

    # Aplicar Factor según la configuración
    if aplicar_factor_antes:
        # Si el factor ya se aplicó antes de normalización, no aplicarlo aquí
        df_con_farmacia['PuntuaciónFinal'] = df_con_farmacia['Puntuación']
    else:
        # Aplicar factor a la puntuación final (comportamiento actual)
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
    df_original, st.session_state.df_coords, df_farmacias, metodo_normalizacion, escala_max, valor_max_personalizado, aplicar_factor_antes
)

df_municipios_farmacias, df_municipios_sin = calcular_puntuaciones(
    df_con_farmacia_base, df_sin_farmacia_base, pesos, radio_km, aplicar_factor_antes
)

# -------------------
# Display ranking table and allow selection
df_ordenado = df_municipios_farmacias.sort_values('PuntuaciónExtendida', ascending=False)
# Deduplicar de forma estable por ID de farmacia si existe, si no por Nombre_Mostrar
if 'ID_Unico' in df_ordenado.columns:
    df_ordenado = df_ordenado.drop_duplicates(subset='ID_Unico', keep='first')
else:
    df_ordenado = df_ordenado.drop_duplicates(subset='Nombre_Mostrar', keep='first')
df_ordenado = df_ordenado.reset_index(drop=True)
df_ordenado.index += 1  # Índice 1-based
# Mostrar información sobre normalización
if metodo_normalizacion != "Sin normalizar":
    if valor_max_personalizado is not None:
        st.info(f"📊 **Normalización aplicada**: {metodo_normalizacion} (escala 0-{escala_max:.0f})")
        st.info(f"🎯 **Rango personalizado**: Máximo establecido en {valor_max_personalizado}")
    else:
        st.info(f"📊 **Normalización aplicada**: {metodo_normalizacion} (escala 0-{escala_max:.0f})")

    if "Logarítmico" in metodo_normalizacion:
        st.info("📈 **Normalización Logarítmica**: Ideal para indicadores con gran disparidad (ej: población, ingresos)")
        st.info("💡 **Beneficio**: Las diferencias en valores bajos son más significativas que en valores altos")

    # Información sobre aplicación del Factor
    if aplicar_factor_antes:
        st.info("⚙️ **Factor aplicado a indicadores individuales** antes de normalización")
    else:
        st.info("⚙️ **Factor aplicado a puntuación final** (comportamiento tradicional)")

    st.info("🎯 **Direccionalidad**: Se controla con pesos positivos/negativos en los sliders")

    # Debug: mostrar algunos valores normalizados
    if len(df_con_farmacia_base) > 0:
        columnas_indicadores = [col for col in df_con_farmacia_base.columns 
                              if col not in ['Territorio', 'Territorio_normalizado', 'Latitud', 'Longitud', 
                                           'Factor', 'Nombre_Mostrar', 'Provincia', 'Ldo']]

        # Mostrar estadísticas de normalización
        with st.expander("📈 Estadísticas de normalización", expanded=False):
            if len(df_con_farmacia_base) > 0:
                # Obtener columnas de indicadores normalizados
                columnas_indicadores = [col for col in df_con_farmacia_base.columns 
                                      if col not in ['Territorio', 'Territorio_normalizado', 'Latitud', 'Longitud', 
                                                   'Factor', 'Nombre_Mostrar', 'Provincia', 'Ldo']]

                if columnas_indicadores:
                    st.write("**Rango de valores normalizados por indicador:**")
                    stats_df = []
                    for col in columnas_indicadores[:10]:  # Mostrar solo los primeros 10
                        if col in df_con_farmacia_base.columns:
                            serie = df_con_farmacia_base[col].dropna()
                            if len(serie) > 0:
                                stats_df.append({
                                    'Indicador': col[:40] + '...' if len(col) > 40 else col,
                                    'Min': f"{serie.min():.2f}",
                                    'Max': f"{serie.max():.2f}",
                                    'Media': f"{serie.mean():.2f}",
                                    'Desv. Est.': f"{serie.std():.2f}"
                                })

                    if stats_df:
                        st.dataframe(pd.DataFrame(stats_df), use_container_width=True)
                    else:
                        st.write("No hay datos normalizados para mostrar.")
                else:
                    st.write("No se encontraron indicadores normalizados.")

    st.subheader("Ranking de municipios con farmacia ordenados por puntuación total")

    # Filtro por provincia
    if 'Provincia' in df_ordenado.columns:
        provincias_disponibles = ['Todas'] + sorted(df_ordenado['Provincia'].dropna().unique().tolist())
        provincia_seleccionada = st.selectbox(
            "Filtrar por provincia:",
            options=provincias_disponibles,
            index=0
        )

        # Aplicar filtro si no se selecciona "Todas"
        if provincia_seleccionada != 'Todas':
            df_ordenado_filtrado = df_ordenado[df_ordenado['Provincia'] == provincia_seleccionada].copy()
            st.info(f"Mostrando {len(df_ordenado_filtrado)} municipios de {provincia_seleccionada}")
        else:
            df_ordenado_filtrado = df_ordenado.copy()
            st.info(f"Mostrando todos los {len(df_ordenado_filtrado)} municipios")
    else:
        df_ordenado_filtrado = df_ordenado.copy()

    if not df_ordenado_filtrado.empty:
        territorio_seleccionado = st.selectbox(
            "Selecciona un municipio del ranking para centrar el mapa:",
            options=df_ordenado_filtrado['Nombre_Mostrar'].tolist()
        )
        
        # -------------------
        # Sistema de GPS - Cálculo de ruta por carretera
        st.markdown("---")
        st.subheader("🗺️ Calculadora de Rutas por Carretera")
        
        # Campo de texto para destino con valor por defecto
        destino_texto = st.text_input(
            "Localidad de destino:",
            value="Granada, España",
            help="Escribe el nombre de la localidad de destino (ej: Granada, España, Sevilla, Málaga, etc.)"
        )
        
        # Botón para calcular ruta
        calcular_ruta = st.button("🚗 Calcular Ruta y Tiempo de Viaje", type="primary")
        
        if calcular_ruta and territorio_seleccionado:
            # Obtener coordenadas del municipio seleccionado
            fila_municipio = df_ordenado_filtrado[df_ordenado_filtrado['Nombre_Mostrar'] == territorio_seleccionado]
            
            if not fila_municipio.empty:
                lat_origen = fila_municipio.iloc[0].get('Latitud')
                lon_origen = fila_municipio.iloc[0].get('Longitud')
                
                if pd.notna(lat_origen) and pd.notna(lon_origen):
                    with st.spinner(f"Calculando ruta desde {territorio_seleccionado} hasta {destino_texto}..."):
                        # Obtener coordenadas del destino
                        lat_destino, lon_destino = obtener_coordenadas_destino(destino_texto)
                        
                        if lat_destino and lon_destino:
                            # Intentar calcular con OSRM primero
                            resultado_osrm = calcular_ruta_osrm(lat_origen, lon_origen, lat_destino, lon_destino)
                            
                            if resultado_osrm:
                                distancia_km, tiempo_minutos, ruta_coordenadas = resultado_osrm
                                metodo_usado = "OSRM (ruta real por carretera)"
                            else:
                                # Usar cálculo estimado como respaldo
                                resultado_estimado = calcular_ruta_estimada(lat_origen, lon_origen, lat_destino, lon_destino)
                                if resultado_estimado:
                                    distancia_km, tiempo_minutos, ruta_coordenadas = resultado_estimado
                                    metodo_usado = "Estimación basada en distancia geodésica"
                                else:
                                    distancia_km, tiempo_minutos, ruta_coordenadas = None, None, None
                                    st.error("No se pudo calcular la distancia. Verifica que las coordenadas sean válidas.")
                            
                            if distancia_km and tiempo_minutos:
                                # Mostrar resultados
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="📍 Distancia",
                                        value=f"{distancia_km:.1f} km"
                                    )
                                
                                with col2:
                                    horas = int(tiempo_minutos // 60)
                                    minutos_restantes = int(tiempo_minutos % 60)
                                    tiempo_formato = f"{horas}h {minutos_restantes}min" if horas > 0 else f"{minutos_restantes}min"
                                    st.metric(
                                        label="⏱️ Tiempo estimado",
                                        value=tiempo_formato
                                    )
                                
                                with col3:
                                    velocidad_promedio = distancia_km / (tiempo_minutos / 60) if tiempo_minutos > 0 else 0
                                    st.metric(
                                        label="🚗 Velocidad promedio",
                                        value=f"{velocidad_promedio:.0f} km/h"
                                    )
                                
                                st.info(f"💡 Método de cálculo: {metodo_usado}")
                                
                                # Guardar información de ruta en session_state para usar en el mapa
                                st.session_state['ruta_calculada'] = {
                                    'origen': territorio_seleccionado,
                                    'destino': destino_texto,
                                    'lat_origen': lat_origen,
                                    'lon_origen': lon_origen,
                                    'lat_destino': lat_destino,
                                    'lon_destino': lon_destino,
                                    'distancia_km': distancia_km,
                                    'tiempo_minutos': tiempo_minutos,
                                    'ruta_coordenadas': ruta_coordenadas
                                }
                            else:
                                st.error("No se pudo calcular la ruta. Por favor, verifica las coordenadas.")
                        else:
                            st.error(f"❌ No se pudieron obtener las coordenadas para: **{destino_texto}**")
                            st.warning("**Posibles soluciones:**")
                            st.markdown("""
                            - Intenta ser más específico, por ejemplo:
                              - 'Granada, España'
                              - 'Sevilla, Andalucía, España'
                              - 'Málaga, Málaga, España'
                            - Verifica la ortografía del nombre de la ciudad
                            - Asegúrate de incluir el país (España) si es necesario
                            - Prueba con el nombre completo de la ciudad
                            """)
                            
                            # Mostrar sugerencias basadas en ciudades conocidas
                            st.info("💡 **Ciudades disponibles en el diccionario:** Granada, Sevilla, Málaga, Córdoba, Cádiz, Jaén, Almería, Huelva, Marbella, Algeciras, Ronda")
                else:
                    st.warning(f"No hay coordenadas disponibles para {territorio_seleccionado}")
            else:
                st.error("No se encontró el municipio seleccionado")
        elif calcular_ruta:
            st.warning("Por favor, selecciona un municipio del ranking primero")
    else:
        territorio_seleccionado = None
        st.info("No hay municipios con farmacia para mostrar en el ranking.")

    # Preparar columnas para mostrar
    columnas_mostrar = ['Ranking', 'Nombre_Mostrar', 'Puntuación', 'Factor', 'PuntuaciónFinal', 'SumaMunicipiosCercanos', 'PuntuaciónExtendida']

    # Añadir Provincia y Población si están disponibles
    if 'Provincia' in df_ordenado.columns:
        columnas_mostrar.insert(2, 'Provincia')  # Insertar después de Nombre_Mostrar

    # Agregar columna de Población al dataframe filtrado
    #st.write(f"Debug: Columnas disponibles en df_ordenado_filtrado: {list(df_ordenado_filtrado.columns)}")

    if 'Territorio' in df_ordenado_filtrado.columns:
        # Calcular población: si hay Singular, tomar población del Territorio y multiplicar por Factor
        df_ordenado_filtrado['Población'] = df_ordenado_filtrado.apply(
            lambda row: obtener_poblacion_territorio_con_factor(
                row['Territorio'],
                row.get('Singular', None) if 'Singular' in df_ordenado_filtrado.columns else None,
                row.get('Factor', None) if 'Factor' in df_ordenado_filtrado.columns else None
            ),
            axis=1
        )
        columnas_mostrar.insert(3, 'Población')  # Insertar después de Provincia
        #st.write("Debug: Columna Población agregada")
    else:
        pass

    # Filtrar solo las columnas que existen
    columnas_existentes = [col for col in columnas_mostrar if col in df_ordenado_filtrado.columns]
    #st.write(f"Debug: Columnas a mostrar: {columnas_mostrar}")
    #st.write(f"Debug: Columnas existentes: {columnas_existentes}")

    st.dataframe(
        df_ordenado_filtrado.reset_index().rename(columns={"index": "Ranking"})[columnas_existentes].round(2),
        use_container_width=True
    )

    # Botón de descarga del ranking con población
    if not df_ordenado_filtrado.empty:
        csv_buffer_ranking = BytesIO()
        df_ranking_export = df_ordenado_filtrado.reset_index().rename(columns={"index": "Ranking"})[columnas_existentes]
        df_ranking_export.to_csv(csv_buffer_ranking, index=False, sep=';', encoding='utf-8')
        csv_buffer_ranking.seek(0)

        st.download_button(
            label="📥 Descargar ranking en CSV",
            data=csv_buffer_ranking,
            file_name="ranking_municipios.csv",
            mime="text/csv"
        )


    # Display detailed breakdown for the selected territory
    if territorio_seleccionado:
        st.subheader(f"Detalle de puntuación para: {territorio_seleccionado}")

        fila_farmacia = df_municipios_farmacias[df_municipios_farmacias["Nombre_Mostrar"] == territorio_seleccionado]
        territorio_original_para_desglose = fila_farmacia.iloc[0]['Territorio'] if not fila_farmacia.empty else None

        if territorio_original_para_desglose:
            # Coincidencia robusta por nombre normalizado e incluir variantes como "(capital)" o "Municipio de ..."
            df_tmp = df_original.copy()
            df_tmp['__terr_norm'] = df_tmp['Territorio'].astype(str).apply(normalizar_nombre_municipio)
            t_base = territorio_original_para_desglose
            targets = {
                normalizar_nombre_municipio(t_base),
                normalizar_nombre_municipio(f"{t_base} (capital)"),
                normalizar_nombre_municipio(f"Municipio de {t_base}"),
            }
            df_territorio = df_tmp[df_tmp['__terr_norm'].isin(targets)].drop(columns=['__terr_norm'])
        else:
            df_territorio = pd.DataFrame()

        if df_territorio.empty:
            st.warning("No hay datos detallados para este territorio.")
            df_desglose = pd.DataFrame()
        else:
            st.write(f"Número de indicadores para {territorio_seleccionado}: ", len(df_territorio))
            desglose = []
            puntuacion_base = 0

            # Obtener valores normalizados del territorio seleccionado
            valores_normalizados = {}
            if not fila_farmacia.empty:
                for col in df_con_farmacia_base.columns:
                    if col not in ['Territorio', 'Territorio_normalizado', 'Latitud', 'Longitud', 
                                 'Factor', 'Nombre_Mostrar', 'Provincia', 'Ldo']:
                        valores_normalizados[col] = fila_farmacia.iloc[0].get(col, 0)

            for _, row in df_territorio.iterrows():
                clave_norm = normaliza_nombre_indicador(row["Medida"])
                valor = row["Valor"]
                peso = pesos.get(clave_norm, 1.0)
                contribucion = valor * peso if pd.notna(valor) else 0
                puntuacion_base += contribucion
                original_display_name = medidas_originales.get(clave_norm, row["Medida"])

                # Obtener valor normalizado si existe
                valor_normalizado = valores_normalizados.get(clave_norm, "N/A")
                if valor_normalizado != "N/A" and pd.notna(valor_normalizado):
                    valor_normalizado = round(valor_normalizado, 2)

                desglose.append({
                    "Indicador": original_display_name,
                    "Valor": round(valor, 2) if pd.notna(valor) else "N/A",
                    "Valor Normalizado": valor_normalizado,
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

    # Ajustar centro si hay ruta calculada
    if 'ruta_calculada' in st.session_state and st.session_state['ruta_calculada']:
        ruta_info = st.session_state['ruta_calculada']
        if ruta_info.get('lat_origen') and ruta_info.get('lat_destino'):
            lat_centro_ruta = (ruta_info['lat_origen'] + ruta_info['lat_destino']) / 2
            lon_centro_ruta = (ruta_info['lon_origen'] + ruta_info['lon_destino']) / 2
            lat_centro = lat_centro_ruta
            lon_centro = lon_centro_ruta
            zoom_nivel = 9
    
    m = folium.Map(location=[lat_centro, lon_centro], zoom_start=zoom_nivel)
    marker_cluster = MarkerCluster().add_to(m)
    
    # Agregar ruta calculada si existe
    if 'ruta_calculada' in st.session_state and st.session_state['ruta_calculada']:
        ruta_info = st.session_state['ruta_calculada']
        
        # Dibujar la ruta si hay coordenadas
        if ruta_info.get('ruta_coordenadas'):
            folium.PolyLine(
                locations=ruta_info['ruta_coordenadas'],
                color='blue',
                weight=4,
                opacity=0.7,
                popup=f"Ruta: {ruta_info['distancia_km']:.1f} km, {ruta_info['tiempo_minutos']:.0f} min"
            ).add_to(m)
        
        # Agregar marcador de origen
        folium.Marker(
            [ruta_info['lat_origen'], ruta_info['lon_origen']],
            popup=f"<b>Origen:</b> {ruta_info['origen']}<br>📍 Inicio de ruta",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        # Agregar marcador de destino
        folium.Marker(
            [ruta_info['lat_destino'], ruta_info['lon_destino']],
            popup=f"<b>Destino:</b> {ruta_info['destino']}<br>🏁 Fin de ruta<br>📏 {ruta_info['distancia_km']:.1f} km<br>⏱️ {ruta_info['tiempo_minutos']:.0f} min",
            icon=folium.Icon(color='red', icon='flag', prefix='fa')
        ).add_to(m)

    # Agrupar filas que comparten las mismas coordenadas (y territorio) para no duplicar marcadores
    if not df_ordenado.empty:
        columnas_group = [col for col in ['Territorio', 'Latitud', 'Longitud'] if col in df_ordenado.columns]
        for (territorio_g, lat, lon), grupo in df_ordenado.groupby(columnas_group):
            # Manejar casos en los que el groupby devuelva tupla distinta si faltan columnas
            if isinstance(territorio_g, (float, int)) and len(columnas_group) == 2:
                # No hay 'Territorio' en el índice del groupby
                lat, lon = territorio_g, lat
                territorio = None
            else:
                territorio = territorio_g

            if pd.isna(lat) or pd.isna(lon):
                continue

            # Color representativo por la puntuación extendida (usar la máxima)
            puntuacion_rep = grupo['PuntuaciónExtendida'].max() if 'PuntuaciónExtendida' in grupo.columns else None
            color = "#777777"
            if puntuacion_rep is not None and not pd.isna(puntuacion_rep):
                for (minv, maxv, col) in rango_colores:
                    if minv <= puntuacion_rep < maxv:
                        color = col
                        break
                if puntuacion_rep >= rango_colores[-1][1]:
                    color = rango_colores[-1][2]

            # Encabezado del popup: usar Territorio si está, si no, el primero de Nombre_Mostrar
            encabezado = str(territorio) if territorio is not None else str(grupo.iloc[0].get('Nombre_Mostrar', 'Entidad'))
            popup_html = f"""
            <b>{encabezado}</b><br>
            """

            # Provincia/Ldo si todos en el grupo coinciden (para no repetir en cada ítem)
            if 'Provincia' in grupo.columns and grupo['Provincia'].notna().any():
                provincias = grupo['Provincia'].dropna().unique().tolist()
                if len(provincias) == 1:
                    popup_html += f"Provincia: {provincias[0]}<br>"
            if 'Ldo' in grupo.columns and grupo['Ldo'].notna().any():
                ldos = grupo['Ldo'].dropna().unique().tolist()
                if len(ldos) == 1:
                    popup_html += f"Ldo: {ldos[0]}<br>"

            # Población del territorio (única por marcador)
            poblacion_original = "N/A"
            try:
                nombre_a_buscar = str(encabezado).strip()
                df_singular_pob = pd.read_csv("singular_pob_sexo.csv", sep=";", na_values=["-", "", "NA"])
                poblacion_data = df_singular_pob[
                    (df_singular_pob['Territorio'] == nombre_a_buscar) &
                    (df_singular_pob['Sexo'] == 'Ambos sexos') &
                    (df_singular_pob['Medida'] == 'Población')
                ]
                if not poblacion_data.empty and pd.notna(poblacion_data.iloc[0]['Valor']):
                    poblacion_original = f"{poblacion_data.iloc[0]['Valor']:.0f}"
            except Exception:
                pass

            popup_html += f"<b>Población:</b> {poblacion_original}<hr style=\"margin:6px 0\">"

            # Listar las entidades que comparten el mismo punto
            popup_html += "<b>Entidades en este punto:</b><br>"
            for _, row in grupo.iterrows():
                nombre_item = str(row.get('Nombre_Mostrar', 'Entidad')).strip()
                factor_item = row.get('Factor', None)
                punt_item = row.get('PuntuaciónFinal', None)
                punt_tot = row.get('PuntuaciónExtendida', None)
                popup_html += (
                    f"- {nombre_item} | Factor: {factor_item:.2f} | "
                    f"PF: {punt_item:.2f} | Total: {punt_tot:.2f}<br>"
                )

            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                popup=folium.Popup(popup_html, max_width=360),
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

        # Convertir a CSV string directamente
        csv_string = df_pesos_guardar.to_csv(index=False, sep=';', encoding='utf-8')

        st.sidebar.download_button(
            label="💾 Descargar configuración actual de pesos",
            data=csv_string,
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

        # Detectar cambios en el ranking para actualizar selección automáticamente
        ranking_actual = df_ranking['Nombre_Mostrar'].head(2).tolist() if len(df_ranking) >= 2 else []
        ranking_anterior = st.session_state.get('ranking_anterior', [])

        # Si el ranking cambió, actualizar la selección por defecto
        if ranking_actual != ranking_anterior and len(ranking_actual) >= 2:
            st.session_state['ranking_anterior'] = ranking_actual
            # Limpiar selecciones anteriores para forzar actualización
            if 'municipio1_selector' in st.session_state:
                del st.session_state['municipio1_selector']
            if 'municipio2_selector' in st.session_state:
                del st.session_state['municipio2_selector']
            st.rerun()

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

                # Obtener valores normalizados para ambos municipios
                valores_normalizados1 = {}
                valores_normalizados2 = {}

                # Buscar filas en df_con_farmacia_base para ambos municipios
                fila_municipio1 = df_con_farmacia_base[df_con_farmacia_base['Nombre_Mostrar'] == municipio1]
                fila_municipio2 = df_con_farmacia_base[df_con_farmacia_base['Nombre_Mostrar'] == municipio2]

                if not fila_municipio1.empty:
                    for col in df_con_farmacia_base.columns:
                        if col not in ['Territorio', 'Territorio_normalizado', 'Latitud', 'Longitud', 
                                     'Factor', 'Nombre_Mostrar', 'Provincia', 'Ldo']:
                            valores_normalizados1[col] = fila_municipio1.iloc[0].get(col, 0)

                if not fila_municipio2.empty:
                    for col in df_con_farmacia_base.columns:
                        if col not in ['Territorio', 'Territorio_normalizado', 'Latitud', 'Longitud', 
                                     'Factor', 'Nombre_Mostrar', 'Provincia', 'Ldo']:
                            valores_normalizados2[col] = fila_municipio2.iloc[0].get(col, 0)

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

                    # Obtener valores normalizados
                    valor_norm1 = valores_normalizados1.get(clave_norm, "N/A")
                    valor_norm2 = valores_normalizados2.get(clave_norm, "N/A")

                    if valor_norm1 != "N/A" and pd.notna(valor_norm1):
                        valor_norm1 = f"{valor_norm1:.2f}"
                    if valor_norm2 != "N/A" and pd.notna(valor_norm2):
                        valor_norm2 = f"{valor_norm2:.2f}"

                    desglose_comparativo.append({
                        'Indicador': indicador,
                        'Peso': f"{peso:.2f}",
                        f'Valor - {municipio1}': f"{valor1:.2f}" if pd.notna(valor1) else "N/A",
                        f'Valor Normalizado - {municipio1}': valor_norm1,
                        f'Contribución - {municipio1}': f"{contribucion1:.2f}",
                        f'Valor - {municipio2}': f"{valor2:.2f}" if pd.notna(valor2) else "N/A",
                        f'Valor Normalizado - {municipio2}': valor_norm2,
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
# TAB 3: Proyecciones Demográficas
with tab3:
    st.header("📈 Proyecciones Demográficas (Entidades singulares)")
    if not motor_entidades_disponible:
        st.error("❌ El motor de entidades singulares no está disponible.")
        st.info("Asegúrate de que 'proyeccion_entidades_singulares_final.py' esté en el directorio raíz.")
    else:
        try:
            render_proyeccion_entidades_singulares()
        except Exception as e:
            st.error(f"❌ Error al renderizar el motor de proyecciones: {e}")

# --------------------
# Version information in the sidebar
st.sidebar.subheader("Version 1.9.1")
