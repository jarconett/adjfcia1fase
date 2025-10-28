"""
Módulo de Proyecciones Demográficas para el Sistema de Farmacias de Andalucía
Autor: Sistema de Análisis Demográfico
Versión: 1.0.0

Este módulo proporciona funcionalidades para calcular proyecciones demográficas
usando datos históricos de crecimiento vegetativo e índices de dependencia.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import unicodedata
warnings.filterwarnings('ignore')

class ProyeccionesDemograficas:
    """
    Clase principal para manejar proyecciones demográficas
    """
    
    def __init__(self):
        self.datos_crecimiento = {}
        self.datos_dependencia = {}
        self.poblacion_actual = {}
        self.tendencias_calculadas = {}
        self._mapa_territorio_a_codigo = None  # cache de Territorios.csv -> código provincia
        # Candidatos de nombre de columna para territorio en ficheros IECA
        self._territorio_column_candidates: List[str] = [
            'Lugar de residencia',
            'Lugar de origen',
            'Lugar de destino',
            'Lugar de procedencia',
            'Territorio',
            'Municipio',
            'Lugar',
        ]

    def _normalizar(self, texto: str) -> str:
        """Normaliza texto: minúsculas y sin acentos/espacios extra."""
        if not isinstance(texto, str):
            return ""
        texto = unicodedata.normalize('NFKD', texto)
        texto = ''.join(c for c in texto if not unicodedata.combining(c))
        return texto.lower().strip()

    def _cargar_mapa_territorios(self) -> Dict[str, str]:
        """Construye un mapa de territorio normalizado -> código de provincia usando Territorios.csv."""
        if self._mapa_territorio_a_codigo is not None:
            return self._mapa_territorio_a_codigo

        # Mapeo de nombre de provincia normalizado a código de fichero
        provincia_a_codigo = {
            'almeria': 'alm',
            'cadiz': 'cad',
            'cordoba': 'cor',
            'granada': 'gra',
            'huelva': 'hue',
            'jaen': 'jae',  # dependencia usa 'jae'; crecimiento tiene 'jaen' (se maneja fallback)
            'malaga': 'mal',
            'sevilla': 'sev',
        }

        try:
            df_terr = pd.read_csv('Territorios.csv', sep=';')
        except Exception:
            # Si no existe, dejar el mapa vacío para usar heurística posterior
            self._mapa_territorio_a_codigo = {}
            return self._mapa_territorio_a_codigo

        mapa: Dict[str, str] = {}
        # Asegurar columnas esperadas
        if 'Territorio' in df_terr.columns and 'Provincia' in df_terr.columns:
            for _, fila in df_terr.iterrows():
                terr_norm = self._normalizar(str(fila['Territorio']))
                prov_norm = self._normalizar(str(fila['Provincia']))
                codigo = provincia_a_codigo.get(prov_norm)
                if terr_norm and codigo:
                    # La última asignación gana; las duplicadas comparten provincia
                    mapa[terr_norm] = codigo

        self._mapa_territorio_a_codigo = mapa
        return self._mapa_territorio_a_codigo

    def _columna_territorio(self, df: pd.DataFrame) -> Optional[str]:
        """Devuelve el nombre real de la columna que representa el territorio.
        Busca entre candidatos conocidos con comparación normalizada.
        """
        if df is None or df.empty:
            return None
        # Mapa de nombre normalizado -> nombre real del DataFrame
        col_norm_to_real: Dict[str, str] = {}
        for col in df.columns:
            col_norm_to_real[self._normalizar(str(col))] = col
        for cand in self._territorio_column_candidates:
            cand_norm = self._normalizar(cand)
            if cand_norm in col_norm_to_real:
                return col_norm_to_real[cand_norm]
        return None
        
    def cargar_datos_crecimiento_vegetativo(self, territorio: str) -> pd.DataFrame:
        """
        Carga datos de crecimiento vegetativo para un territorio específico
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            DataFrame con datos de crecimiento vegetativo
        """
        try:
            # Determinar provincia del territorio
            provincia = self._determinar_provincia(territorio)
            
            # Cargar archivo de crecimiento vegetativo
            archivo_crecimiento = f"demografia/ieca_export_crec_veg_{provincia}.csv"
            
            # Manejar casos especiales (Granada tiene dos archivos)
            if provincia == "gra":
                # Intentar cargar ambos archivos y concatenarlos
                try:
                    df1 = pd.read_csv("demografia/ieca_export_crec_veg_gra1.csv", sep=";", decimal=",")
                    df2 = pd.read_csv("demografia/ieca_export_crec_veg_gra2.csv", sep=";", decimal=",")
                    df_crecimiento = pd.concat([df1, df2], ignore_index=True)
                except FileNotFoundError:
                    # Si no existen archivos separados, usar el archivo único
                    df_crecimiento = pd.read_csv(archivo_crecimiento, sep=";", decimal=",")
            else:
                # Intentar lectura normal y aplicar fallback para Jaén (jae -> jaen)
                try:
                    df_crecimiento = pd.read_csv(archivo_crecimiento, sep=";", decimal=",")
                except FileNotFoundError:
                    if provincia == "jae":
                        archivo_crecimiento_alt = "demografia/ieca_export_crec_veg_jaen.csv"
                        df_crecimiento = pd.read_csv(archivo_crecimiento_alt, sep=";", decimal=",")
                    else:
                        raise
            
            # Filtrar por territorio (comparación robusta normalizada)
            terr_norm = self._normalizar(territorio)
            df_crecimiento = df_crecimiento.copy()
            col_terr = self._columna_territorio(df_crecimiento)
            if not col_terr:
                st.error("No se encontró columna de territorio en datos de crecimiento")
                return pd.DataFrame()
            df_crecimiento['__terr_norm'] = df_crecimiento[col_terr].astype(str).map(self._normalizar)
            df_territorio = df_crecimiento[df_crecimiento['__terr_norm'] == terr_norm].drop(columns=['__terr_norm']).copy()
            
            # Convertir columna Anual a numérico
            df_territorio['Anual'] = pd.to_numeric(df_territorio['Anual'], errors='coerce')
            df_territorio['Valor'] = pd.to_numeric(df_territorio['Valor'], errors='coerce')
            
            # Eliminar filas con valores nulos
            df_territorio = df_territorio.dropna(subset=['Anual', 'Valor'])
            
            return df_territorio
            
        except FileNotFoundError:
            st.error(f"No se encontró el archivo de crecimiento vegetativo para {territorio}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error al cargar datos de crecimiento vegetativo: {e}")
            return pd.DataFrame()
    
    def cargar_datos_dependencia(self, territorio: str) -> pd.DataFrame:
        """
        Carga datos de índices de dependencia para un territorio específico
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            DataFrame con datos de dependencia
        """
        try:
            # Determinar provincia del territorio
            provincia = self._determinar_provincia(territorio)
            
            # Cargar ambos archivos de dependencia
            archivo_dep1 = f"demografia/ieca_export_dep_{provincia}1.csv"
            archivo_dep2 = f"demografia/ieca_export_dep_{provincia}2.csv"
            
            df_dep1 = pd.read_csv(archivo_dep1, sep=";", decimal=",")
            df_dep2 = pd.read_csv(archivo_dep2, sep=";", decimal=",")
            
            # Concatenar ambos archivos
            df_dependencia = pd.concat([df_dep1, df_dep2], ignore_index=True)
            
            # Filtrar por territorio (comparación robusta normalizada)
            terr_norm = self._normalizar(territorio)
            df_dependencia = df_dependencia.copy()
            col_terr = self._columna_territorio(df_dependencia)
            if not col_terr:
                st.error("No se encontró columna de territorio en datos de dependencia")
                return pd.DataFrame()
            df_dependencia['__terr_norm'] = df_dependencia[col_terr].astype(str).map(self._normalizar)
            df_territorio = df_dependencia[df_dependencia['__terr_norm'] == terr_norm].drop(columns=['__terr_norm']).copy()
            
            # Convertir columnas a numérico
            df_territorio['Anual'] = pd.to_numeric(df_territorio['Anual'], errors='coerce')
            df_territorio['Valor'] = pd.to_numeric(df_territorio['Valor'], errors='coerce')
            
            # Eliminar filas con valores nulos
            df_territorio = df_territorio.dropna(subset=['Anual', 'Valor'])
            
            return df_territorio
            
        except FileNotFoundError:
            st.error(f"No se encontraron archivos de dependencia para {territorio}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error al cargar datos de dependencia: {e}")
            return pd.DataFrame()
    
    def cargar_datos_emigracion(self, territorio: str) -> pd.DataFrame:
        """
        Carga datos de emigración por edad y sexo para un territorio específico
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            DataFrame con datos de emigración filtrados por territorio
        """
        try:
            provincia = self._determinar_provincia(territorio)
            
            # Construir nombre del archivo de emigración
            archivo_emigracion = f"demografia/ieca_export_emigraciones_edad_sexo_88_09_{provincia}.csv"
            
            # Manejar casos especiales (Granada tiene dos archivos)
            if provincia == "gra":
                # Intentar cargar ambos archivos y concatenarlos
                try:
                    df1 = pd.read_csv("demografia/ieca_export_emigraciones_edad_sexo_88_09_gra1.csv", sep=";", decimal=",")
                    df2 = pd.read_csv("demografia/ieca_export_emigraciones_edad_sexo_88_09_gra2.csv", sep=";", decimal=",")
                    df_emigracion = pd.concat([df1, df2], ignore_index=True)
                except FileNotFoundError:
                    # Si no existen archivos separados, usar el archivo único
                    df_emigracion = pd.read_csv(archivo_emigracion, sep=";", decimal=",")
            else:
                df_emigracion = pd.read_csv(archivo_emigracion, sep=";", decimal=",")
            
            # Filtrar por territorio (comparación robusta normalizada)
            terr_norm = self._normalizar(territorio)
            df_emigracion = df_emigracion.copy()
            col_terr = self._columna_territorio(df_emigracion)
            if not col_terr:
                st.error("No se encontró columna de territorio en datos de emigración")
                return pd.DataFrame()
            df_emigracion['__terr_norm'] = df_emigracion[col_terr].astype(str).map(self._normalizar)
            df_territorio = df_emigracion[df_emigracion['__terr_norm'] == terr_norm].drop(columns=['__terr_norm']).copy()
            
            # Convertir columnas a numérico
            df_territorio['Anual'] = pd.to_numeric(df_territorio['Anual'], errors='coerce')
            df_territorio['Valor'] = pd.to_numeric(df_territorio['Valor'], errors='coerce')
            
            # Filtrar valores válidos
            df_territorio = df_territorio.dropna(subset=['Anual', 'Valor'])
            
            return df_territorio
            
        except FileNotFoundError:
            st.warning(f"No se encontró archivo de emigración para {territorio}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error al cargar datos de emigración: {e}")
            return pd.DataFrame()
    
    def cargar_datos_inmigracion(self, territorio: str) -> pd.DataFrame:
        """
        Carga datos de inmigración por edad y sexo para un territorio específico
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            DataFrame con datos de inmigración filtrados por territorio
        """
        try:
            provincia = self._determinar_provincia(territorio)
            
            # Construir nombre del archivo de inmigración
            archivo_inmigracion = f"demografia/ieca_export_inmigraciones_edad_sexo_88_09_{provincia}.csv"
            
            # Manejar casos especiales (Granada tiene dos archivos)
            if provincia == "gra":
                # Intentar cargar ambos archivos y concatenarlos
                try:
                    df1 = pd.read_csv("demografia/ieca_export_inmigraciones_edad_sexo_88_09_gra1.csv", sep=";", decimal=",")
                    df2 = pd.read_csv("demografia/ieca_export_inmigraciones_edad_sexo_88_09_gra2.csv", sep=";", decimal=",")
                    df_inmigracion = pd.concat([df1, df2], ignore_index=True)
                except FileNotFoundError:
                    # Si no existen archivos separados, usar el archivo único
                    df_inmigracion = pd.read_csv(archivo_inmigracion, sep=";", decimal=",")
            else:
                df_inmigracion = pd.read_csv(archivo_inmigracion, sep=";", decimal=",")
            
            # Filtrar por territorio (comparación robusta normalizada)
            terr_norm = self._normalizar(territorio)
            df_inmigracion = df_inmigracion.copy()
            col_terr = self._columna_territorio(df_inmigracion)
            if not col_terr:
                st.error("No se encontró columna de territorio en datos de inmigración")
                return pd.DataFrame()
            df_inmigracion['__terr_norm'] = df_inmigracion[col_terr].astype(str).map(self._normalizar)
            df_territorio = df_inmigracion[df_inmigracion['__terr_norm'] == terr_norm].drop(columns=['__terr_norm']).copy()
            
            # Convertir columnas a numérico
            df_territorio['Anual'] = pd.to_numeric(df_territorio['Anual'], errors='coerce')
            df_territorio['Valor'] = pd.to_numeric(df_territorio['Valor'], errors='coerce')
            
            # Filtrar valores válidos
            df_territorio = df_territorio.dropna(subset=['Anual', 'Valor'])
            
            return df_territorio
            
        except FileNotFoundError:
            st.warning(f"No se encontró archivo de inmigración para {territorio}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error al cargar datos de inmigración: {e}")
            return pd.DataFrame()
    
    def cargar_datos_paro(self, territorio: str) -> pd.DataFrame:
        """
        Carga datos de paro por edad y sexo para un territorio específico
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            DataFrame con datos de paro filtrados por territorio
        """
        try:
            # Los archivos de paro tienen diferentes rangos de años
            archivos_paro = [
                "demografia/ieca_export_paro_07-09.csv",
                "demografia/ieca_export_paro_10-12.csv", 
                "demografia/ieca_export_paro_13-15.csv",
                "demografia/ieca_export_paro_16-18.csv",
                "demografia/ieca_export_paro_19-21.csv",
                "demografia/ieca_export_paro_22-24.csv"
            ]
            
            dfs_paro = []
            for archivo in archivos_paro:
                try:
                    df_temp = pd.read_csv(archivo, sep=";", decimal=",")
                    dfs_paro.append(df_temp)
                except FileNotFoundError:
                    continue
            
            if not dfs_paro:
                st.warning(f"No se encontraron archivos de paro")
                return pd.DataFrame()
            
            # Concatenar todos los archivos disponibles
            df_paro = pd.concat(dfs_paro, ignore_index=True)
            
            # Filtrar por territorio (comparación robusta normalizada)
            terr_norm = self._normalizar(territorio)
            df_paro = df_paro.copy()
            col_terr = self._columna_territorio(df_paro)
            if not col_terr:
                st.error("No se encontró columna de territorio en datos de paro")
                return pd.DataFrame()
            df_paro['__terr_norm'] = df_paro[col_terr].astype(str).map(self._normalizar)
            df_territorio = df_paro[df_paro['__terr_norm'] == terr_norm].drop(columns=['__terr_norm']).copy()
            
            # Convertir columnas a numérico
            df_territorio['Anual'] = pd.to_numeric(df_territorio['Anual'], errors='coerce')
            df_territorio['Valor'] = pd.to_numeric(df_territorio['Valor'], errors='coerce')
            
            # Filtrar valores válidos
            df_territorio = df_territorio.dropna(subset=['Anual', 'Valor'])
            
            return df_territorio
            
        except Exception as e:
            st.error(f"Error al cargar datos de paro: {e}")
            return pd.DataFrame()
    
    def _determinar_provincia(self, territorio: str) -> str:
        """
        Determina el código de provincia (3 letras) para un territorio usando `Territorios.csv`.
        Fallback: si no se encuentra, usa heurísticas y por defecto primeros 3 caracteres.
        """
        # Soportar nombres de provincias directamente (con o sin acento)
        provincia_a_codigo = {
            'almeria': 'alm', 'almería': 'alm',
            'cadiz': 'cad', 'cádiz': 'cad',
            'cordoba': 'cor', 'córdoba': 'cor',
            'granada': 'gra',
            'huelva': 'hue',
            'jaen': 'jae', 'jaén': 'jae',
            'malaga': 'mal', 'málaga': 'mal',
            'sevilla': 'sev',
        }

        terr_norm = self._normalizar(territorio)

        # Coincidencia directa con nombre de provincia
        if terr_norm in provincia_a_codigo:
            return provincia_a_codigo[terr_norm]

        # Buscar en el mapeo de Territorios.csv
        mapa = self._cargar_mapa_territorios()
        codigo = mapa.get(terr_norm)
        if codigo:
            return codigo

        # Heurística: si el nombre del territorio contiene el nombre de provincia
        for prov_nombre, cod in provincia_a_codigo.items():
            if terr_norm in prov_nombre or prov_nombre in terr_norm:
                return cod

        # Por defecto, asumir que es el territorio mismo (nombre corto)
        return terr_norm[:3]
    
    def obtener_territorios_con_farmacia(self, df_farmacias: pd.DataFrame) -> List[str]:
        """
        Obtiene la lista de territorios que tienen farmacia
        
        Args:
            df_farmacias: DataFrame con datos de farmacias
            
        Returns:
            Lista de territorios con farmacia
        """
        if df_farmacias.empty or 'Territorio' not in df_farmacias.columns:
            return []
        
        territorios_farmacia = df_farmacias['Territorio'].dropna().unique().tolist()
        return sorted(territorios_farmacia)
    
    def verificar_territorio_tiene_datos_demograficos(self, territorio: str) -> bool:
        """
        Verifica si un territorio tiene datos demográficos disponibles
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            True si tiene datos, False en caso contrario
        """
        try:
            datos_crecimiento = self.cargar_datos_crecimiento_vegetativo(territorio)
            datos_dependencia = self.cargar_datos_dependencia(territorio)
            
            return not datos_crecimiento.empty and not datos_dependencia.empty
        except:
            return False
    
    def analizar_tendencias_demograficas(self, territorio: str) -> Dict:
        """
        Analiza las tendencias históricas para un territorio específico incluyendo migración y paro
        
        Args:
            territorio: Nombre del territorio
            
        Returns:
            Diccionario con tendencias calculadas
        """
        # Cargar datos históricos
        datos_crecimiento = self.cargar_datos_crecimiento_vegetativo(territorio)
        datos_dependencia = self.cargar_datos_dependencia(territorio)
        datos_emigracion = self.cargar_datos_emigracion(territorio)
        datos_inmigracion = self.cargar_datos_inmigracion(territorio)
        datos_paro = self.cargar_datos_paro(territorio)
        
        if datos_crecimiento.empty or datos_dependencia.empty:
            return {}
        
        tendencias = {}
        
        # 1. Análisis del crecimiento vegetativo
        tendencias['crecimiento'] = self._calcular_tendencias_crecimiento(datos_crecimiento)
        
        # 2. Análisis de índices de dependencia
        tendencias['dependencia'] = self._calcular_tendencias_dependencia(datos_dependencia)
        
        # 3. Análisis de migración si están disponibles
        if not datos_emigracion.empty and not datos_inmigracion.empty:
            tendencias['migracion'] = self._analizar_tendencias_migracion(datos_emigracion, datos_inmigracion)
        
        # 4. Análisis de paro si están disponibles
        if not datos_paro.empty:
            tendencias['paro'] = self._analizar_tendencias_paro(datos_paro)
        
        # 5. Detectar puntos de inflexión
        tendencias['puntos_inflexion'] = self._detectar_cambios_tendencia(datos_crecimiento)
        
        # 6. Calcular estadísticas descriptivas
        tendencias['estadisticas'] = self._calcular_estadisticas_descriptivas(
            datos_crecimiento, datos_dependencia
        )
        
        return tendencias
    
    def _calcular_tendencias_crecimiento(self, datos: pd.DataFrame) -> Dict:
        """
        Calcula tendencias de crecimiento vegetativo por sexo
        """
        tendencias = {}
        
        for sexo in ['Ambos sexos', 'Hombres', 'Mujeres']:
            datos_sexo = datos[datos['Sexo'] == sexo].copy()
            
            if not datos_sexo.empty:
                # Regresión lineal simple
                x = datos_sexo['Anual'].values
                y = datos_sexo['Valor'].values
                
                # Calcular pendiente e intercepto
                pendiente, intercepto = np.polyfit(x, y, 1)
                
                # Calcular R²
                y_pred = pendiente * x + intercepto
                r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
                
                # Calcular tasa de crecimiento promedio
                tasa_crecimiento_promedio = np.mean(np.diff(y) / y[:-1]) * 100
                
                tendencias[sexo.lower().replace(' ', '_')] = {
                    'pendiente': pendiente,
                    'intercepto': intercepto,
                    'r_squared': r_squared,
                    'tasa_crecimiento_promedio': tasa_crecimiento_promedio,
                    'valor_ultimo': y[-1],
                    'año_ultimo': x[-1],
                    'valor_primer': y[0],
                    'año_primer': x[0]
                }
        
        return tendencias
    
    def _calcular_tendencia_lineal(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calcula la regresión lineal simple para un conjunto de datos
        
        Args:
            x: Valores del eje X (años)
            y: Valores del eje Y (valores)
            
        Returns:
            Diccionario con pendiente, intercepto y estadísticas
        """
        if len(x) < 2 or len(y) < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}
        
        # Calcular pendiente e intercepto
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calcular R²
        y_pred = slope * x + intercept
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        }
    
    def _analizar_tendencias_migracion(self, datos_emigracion: pd.DataFrame, datos_inmigracion: pd.DataFrame) -> Dict:
        """
        Analiza tendencias de migración neta por edad y sexo
        """
        tendencias = {}
        
        # Calcular migración neta por año y sexo
        migracion_neta = {}
        
        for sexo in ['Ambos sexos', 'Hombres', 'Mujeres']:
            emigracion_sexo = datos_emigracion[datos_emigracion['Sexo'] == sexo]
            inmigracion_sexo = datos_inmigracion[datos_inmigracion['Sexo'] == sexo]
            
            if not emigracion_sexo.empty and not inmigracion_sexo.empty:
                # Agrupar por año y calcular totales
                emigracion_anual = emigracion_sexo.groupby('Anual')['Valor'].sum()
                inmigracion_anual = inmigracion_sexo.groupby('Anual')['Valor'].sum()
                
                # Calcular migración neta (inmigración - emigración)
                años_comunes = emigracion_anual.index.intersection(inmigracion_anual.index)
                migracion_neta_sexo = inmigracion_anual[años_comunes] - emigracion_anual[años_comunes]
                
                if len(migracion_neta_sexo) > 1:
                    # Calcular tendencia de migración neta
                    x = migracion_neta_sexo.index.values
                    y = migracion_neta_sexo.values
                    
                    tendencia = self._calcular_tendencia_lineal(x, y)
                    tendencias[sexo.lower().replace(' ', '_')] = tendencia
                    
                    # Calcular estadísticas adicionales
                    tendencias[f"{sexo.lower().replace(' ', '_')}_estadisticas"] = {
                        'migracion_neta_promedio': migracion_neta_sexo.mean(),
                        'migracion_neta_ultimo_año': migracion_neta_sexo.iloc[-1] if len(migracion_neta_sexo) > 0 else 0,
                        'años_datos': len(migracion_neta_sexo)
                    }
        
        return tendencias
    
    def _analizar_tendencias_paro(self, datos_paro: pd.DataFrame) -> Dict:
        """
        Analiza tendencias de paro por edad y sexo
        """
        tendencias = {}
        
        # Filtrar solo datos de demandantes de empleo
        datos_demandantes = datos_paro[datos_paro['Medida'] == 'Demandantes']
        
        if datos_demandantes.empty:
            return tendencias
        
        for sexo in ['Ambos sexos', 'Hombres', 'Mujeres']:
            datos_sexo = datos_demandantes[datos_demandantes['Sexo'] == sexo]
            
            if not datos_sexo.empty:
                # Agrupar por año y calcular totales
                paro_anual = datos_sexo.groupby('Anual')['Valor'].sum()
                
                if len(paro_anual) > 1:
                    # Calcular tendencia de paro
                    x = paro_anual.index.values
                    y = paro_anual.values
                    
                    tendencia = self._calcular_tendencia_lineal(x, y)
                    tendencias[sexo.lower().replace(' ', '_')] = tendencia
                    
                    # Calcular estadísticas adicionales
                    tendencias[f"{sexo.lower().replace(' ', '_')}_estadisticas"] = {
                        'paro_promedio': paro_anual.mean(),
                        'paro_ultimo_año': paro_anual.iloc[-1] if len(paro_anual) > 0 else 0,
                        'años_datos': len(paro_anual),
                        'tendencia_paro': 'creciente' if tendencia['slope'] > 0 else 'decreciente'
                    }
        
        return tendencias
    
    def _calcular_tendencias_dependencia(self, datos: pd.DataFrame) -> Dict:
        """
        Calcula tendencias de índices de dependencia
        """
        tendencias = {}
        
        tipos_dependencia = {
            'Índice de dependencia global': 'global',
            'Índice de dependencia jóvenes': 'jovenes',
            'Índice de dependencia mayores': 'mayores'
        }
        
        for tipo_original, tipo_clave in tipos_dependencia.items():
            datos_tipo = datos[datos['Edad'] == tipo_original].copy()
            
            if not datos_tipo.empty:
                x = datos_tipo['Anual'].values
                y = datos_tipo['Valor'].values
                
                # Regresión lineal
                pendiente, intercepto = np.polyfit(x, y, 1)
                
                # Calcular R²
                y_pred = pendiente * x + intercepto
                r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
                
                # Calcular cambio promedio anual
                cambio_anual_promedio = pendiente
                
                tendencias[tipo_clave] = {
                    'pendiente': pendiente,
                    'intercepto': intercepto,
                    'r_squared': r_squared,
                    'cambio_anual_promedio': cambio_anual_promedio,
                    'valor_ultimo': y[-1],
                    'año_ultimo': x[-1],
                    'valor_primer': y[0],
                    'año_primer': x[0]
                }
        
        return tendencias
    
    def _detectar_cambios_tendencia(self, datos: pd.DataFrame) -> List[Dict]:
        """
        Detecta puntos de inflexión en las tendencias
        """
        puntos_inflexion = []
        
        # Usar datos de "Ambos sexos" para detectar cambios
        datos_ambos = datos[datos['Sexo'] == 'Ambos sexos'].copy()
        
        if len(datos_ambos) < 5:  # Necesitamos suficientes puntos
            return puntos_inflexion
        
        # Calcular diferencias de segundo orden para detectar cambios en la aceleración
        valores = datos_ambos['Valor'].values
        años = datos_ambos['Anual'].values
        
        # Calcular segunda derivada (aproximada)
        segunda_derivada = np.diff(valores, 2)
        
        # Detectar cambios significativos en la segunda derivada
        umbral = np.std(segunda_derivada) * 1.5
        
        for i, cambio in enumerate(segunda_derivada):
            if abs(cambio) > umbral:
                año_inflexion = años[i + 2]  # Ajustar índice
                puntos_inflexion.append({
                    'año': año_inflexion,
                    'cambio': cambio,
                    'tipo': 'aceleracion' if cambio > 0 else 'desaceleracion'
                })
        
        return puntos_inflexion
    
    def _calcular_estadisticas_descriptivas(self, datos_crecimiento: pd.DataFrame, 
                                          datos_dependencia: pd.DataFrame) -> Dict:
        """
        Calcula estadísticas descriptivas de los datos históricos
        """
        estadisticas = {}
        
        # Estadísticas de crecimiento vegetativo
        datos_ambos = datos_crecimiento[datos_crecimiento['Sexo'] == 'Ambos sexos']
        if not datos_ambos.empty:
            valores_crecimiento = datos_ambos['Valor'].values
            estadisticas['crecimiento'] = {
                'media': np.mean(valores_crecimiento),
                'mediana': np.median(valores_crecimiento),
                'desviacion_estandar': np.std(valores_crecimiento),
                'minimo': np.min(valores_crecimiento),
                'maximo': np.max(valores_crecimiento),
                'coeficiente_variacion': np.std(valores_crecimiento) / np.mean(valores_crecimiento) * 100
            }
        
        # Estadísticas de dependencia
        for tipo in ['Índice de dependencia global', 'Índice de dependencia jóvenes', 'Índice de dependencia mayores']:
            datos_tipo = datos_dependencia[datos_dependencia['Edad'] == tipo]
            if not datos_tipo.empty:
                valores_dep = datos_tipo['Valor'].values
                clave = tipo.split()[-1].lower()  # 'global', 'jóvenes', 'mayores'
                estadisticas[f'dependencia_{clave}'] = {
                    'media': np.mean(valores_dep),
                    'mediana': np.median(valores_dep),
                    'desviacion_estandar': np.std(valores_dep),
                    'minimo': np.min(valores_dep),
                    'maximo': np.max(valores_dep)
                }
        
        return estadisticas
    
    def proyectar_demografia(self, territorio: str, poblacion_actual: float, 
                           años_proyeccion: int, modelo: str = "lineal") -> Dict:
        """
        Proyecta la demografía usando el modelo especificado
        
        Args:
            territorio: Nombre del territorio
            poblacion_actual: Población actual del territorio
            años_proyeccion: Número de años a proyectar
            modelo: Tipo de modelo a usar
            
        Returns:
            Diccionario con proyecciones calculadas
        """
        # Obtener tendencias
        tendencias = self.analizar_tendencias_demograficas(territorio)
        
        if not tendencias:
            return {}
        
        proyecciones = {}
        
        # Proyectar según el modelo seleccionado
        if modelo == "lineal":
            proyecciones = self._proyectar_tendencia_lineal(
                poblacion_actual, tendencias, años_proyeccion
            )
        elif modelo == "exponencial":
            proyecciones = self._proyectar_tendencia_exponencial(
                poblacion_actual, tendencias, años_proyeccion
            )
        elif modelo == "componentes":
            proyecciones = self._proyectar_por_componentes(
                poblacion_actual, tendencias, años_proyeccion
            )
        elif modelo == "comparar_todos":
            proyecciones = self._proyectar_todos_modelos(
                poblacion_actual, tendencias, años_proyeccion
            )
        
        return {
            'proyecciones': proyecciones,
            'tendencias': tendencias,
            'territorio': territorio,
            'años_proyeccion': años_proyeccion,
            'modelo': modelo
        }
    
    def _proyectar_tendencia_lineal(self, poblacion_actual: float, 
                                  tendencias: Dict, años: int) -> Dict:
        """
        Proyección usando tendencia lineal
        """
        crecimiento = tendencias['crecimiento']['ambos_sexos']
        
        # Usar la pendiente para proyectar crecimiento vegetativo
        pendiente = crecimiento['pendiente']
        intercepto = crecimiento['intercepto']
        año_actual = crecimiento['año_ultimo']
        
        proyecciones = {}
        
        for año in range(1, años + 1):
            año_proyeccion = año_actual + año
            
            # Proyectar crecimiento vegetativo
            crecimiento_proyectado = pendiente * año_proyeccion + intercepto
            
            # Proyectar población (asumiendo que el crecimiento vegetativo es proporcional)
            # Esto es una simplificación - en realidad necesitaríamos más datos
            tasa_crecimiento_poblacion = crecimiento_proyectado / poblacion_actual
            poblacion_proyectada = poblacion_actual * (1 + tasa_crecimiento_poblacion) ** año
            
            proyecciones[año] = {
                'año': año_proyeccion,
                'poblacion_total': poblacion_proyectada,
                'crecimiento_vegetativo': crecimiento_proyectado,
                'tasa_crecimiento': tasa_crecimiento_poblacion * 100
            }
        
        return proyecciones
    
    def _proyectar_tendencia_exponencial(self, poblacion_actual: float, 
                                       tendencias: Dict, años: int) -> Dict:
        """
        Proyección usando tendencia exponencial
        """
        crecimiento = tendencias['crecimiento']['ambos_sexos']
        tasa_crecimiento_promedio = crecimiento['tasa_crecimiento_promedio'] / 100
        
        proyecciones = {}
        
        for año in range(1, años + 1):
            año_proyeccion = crecimiento['año_ultimo'] + año
            
            # Proyección exponencial
            poblacion_proyectada = poblacion_actual * (1 + tasa_crecimiento_promedio) ** año
            
            # Estimar crecimiento vegetativo proporcional
            crecimiento_proyectado = poblacion_proyectada * tasa_crecimiento_promedio
            
            proyecciones[año] = {
                'año': año_proyeccion,
                'poblacion_total': poblacion_proyectada,
                'crecimiento_vegetativo': crecimiento_proyectado,
                'tasa_crecimiento': tasa_crecimiento_promedio * 100
            }
        
        return proyecciones
    
    def _proyectar_por_componentes(self, poblacion_actual: float, 
                                 tendencias: Dict, años: int) -> Dict:
        """
        Proyección desagregando por componentes demográficos incluyendo migración neta
        """
        proyecciones = {}
        
        # Usar tendencias de dependencia para estimar cambios en la estructura
        dependencia_global = tendencias['dependencia']['global']
        dependencia_mayores = tendencias['dependencia']['mayores']
        
        # Obtener tendencias de migración si están disponibles
        migracion_ambos = tendencias.get('migracion', {}).get('ambos_sexos', {})
        
        # Calcular población inicial por grupos de edad (estimación)
        poblacion_activa = poblacion_actual * 0.65  # 15-64 años
        poblacion_dependiente = poblacion_actual * 0.35  # 0-14 + 65+ años
        
        for año in range(1, años + 1):
            año_proyeccion = dependencia_global['año_ultimo'] + año
            
            # Proyectar índices de dependencia
            indice_dep_global = dependencia_global['pendiente'] * año_proyeccion + dependencia_global['intercepto']
            indice_dep_mayores = dependencia_mayores['pendiente'] * año_proyeccion + dependencia_mayores['intercepto']
            
            # Proyectar migración neta si está disponible
            migracion_neta = 0
            if migracion_ambos:
                migracion_neta = migracion_ambos.get('slope', 0) * año + migracion_ambos.get('intercept', 0)
            
            # Aplicar migración neta a la población activa (mayor impacto)
            poblacion_activa_proy = poblacion_activa + migracion_neta * 0.7
            
            # Población dependiente total basada en índice de dependencia
            poblacion_dependiente_proy = poblacion_activa_proy * (indice_dep_global / 100)
            
            # Población mayor (65+) basada en índice de dependencia de mayores
            poblacion_mayores_proy = poblacion_activa_proy * (indice_dep_mayores / 100)
            
            # Población joven (0-14)
            poblacion_jovenes_proy = poblacion_dependiente_proy - poblacion_mayores_proy
            
            # Población total
            poblacion_total = poblacion_activa_proy + poblacion_dependiente_proy
            
            # Calcular tasa de crecimiento total (incluyendo migración)
            tasa_crecimiento_total = migracion_neta / poblacion_total * 100 if poblacion_total > 0 else 0
            
            proyecciones[año] = {
                'año': año_proyeccion,
                'poblacion_total': poblacion_total,
                'poblacion_activa': poblacion_activa_proy,
                'poblacion_jovenes': poblacion_jovenes_proy,
                'poblacion_mayores': poblacion_mayores_proy,
                'migracion_neta': migracion_neta,
                'tasa_crecimiento': tasa_crecimiento_total,
                'indice_dependencia_global': indice_dep_global,
                'indice_dependencia_mayores': indice_dep_mayores,
                'indice_dependencia_jovenes': (poblacion_jovenes_proy / poblacion_activa_proy) * 100 if poblacion_activa_proy > 0 else 0
            }
            
            # Actualizar poblaciones para el siguiente año
            poblacion_activa = poblacion_activa_proy
            poblacion_dependiente = poblacion_dependiente_proy
        
        return proyecciones
    
    def _proyectar_todos_modelos(self, poblacion_actual: float, 
                               tendencias: Dict, años: int) -> Dict:
        """
        Proyecta usando todos los modelos para comparación
        """
        modelos = ['lineal', 'exponencial', 'componentes']
        proyecciones_todos = {}
        
        for modelo in modelos:
            if modelo == 'lineal':
                proyecciones_todos[modelo] = self._proyectar_tendencia_lineal(
                    poblacion_actual, tendencias, años
                )
            elif modelo == 'exponencial':
                proyecciones_todos[modelo] = self._proyectar_tendencia_exponencial(
                    poblacion_actual, tendencias, años
                )
            elif modelo == 'componentes':
                proyecciones_todos[modelo] = self._proyectar_por_componentes(
                    poblacion_actual, tendencias, años
                )
        
        return proyecciones_todos
    
    def generar_graficos_proyeccion(self, resultado: Dict) -> Dict:
        """
        Genera gráficos para visualizar las proyecciones con alertas visuales
        """
        graficos = {}
        
        proyecciones = resultado['proyecciones']
        territorio = resultado['territorio']
        modelo = resultado['modelo']
        indicadores = resultado['indicadores']
        
        if modelo == 'comparar_todos':
            # Gráfico comparativo de todos los modelos
            fig = go.Figure()
            
            for nombre_modelo, datos_modelo in proyecciones.items():
                años = [datos_modelo[año]['año'] for año in datos_modelo.keys()]
                poblaciones = [datos_modelo[año]['poblacion_total'] for año in datos_modelo.keys()]
                
                # Obtener colores según indicadores
                indicadores_modelo = indicadores[nombre_modelo]
                color_linea = indicadores_modelo['riesgo_despoblacion_color']
                
                fig.add_trace(go.Scatter(
                    x=años,
                    y=poblaciones,
                    mode='lines+markers',
                    name=f'Modelo {nombre_modelo.title()}',
                    line=dict(width=3, color=color_linea),
                    marker=dict(size=6)
                ))
            
            # Añadir línea de referencia de 1000 habitantes
            fig.add_hline(
                y=1000, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Umbral 1000 habitantes",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f'Comparación de Proyecciones Demográficas - {territorio}',
                xaxis_title='Año',
                yaxis_title='Población Total',
                hovermode='x unified',
                height=500
            )
            
            graficos['comparativo'] = fig
        
        else:
            # Gráfico simple para un modelo
            años = [proyecciones[año]['año'] for año in proyecciones.keys()]
            poblaciones = [proyecciones[año]['poblacion_total'] for año in proyecciones.keys()]
            
            fig = go.Figure()
            
            # Color de línea según riesgo de despoblación
            color_linea = indicadores['riesgo_despoblacion_color']
            
            fig.add_trace(go.Scatter(
                x=años,
                y=poblaciones,
                mode='lines+markers',
                name=f'Proyección {modelo.title()}',
                line=dict(width=4, color=color_linea),
                marker=dict(size=8)
            ))
            
            # Añadir línea de referencia de 1000 habitantes
            fig.add_hline(
                y=1000, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Umbral 1000 habitantes",
                annotation_position="top right"
            )
            
            # Añadir anotaciones según indicadores
            poblacion_final = poblaciones[-1]
            año_final = años[-1]
            
            if indicadores['puede_superar_1000']:
                fig.add_annotation(
                    x=año_final,
                    y=poblacion_final,
                    text=f"✅ Supera 1000 hab<br>({poblacion_final:,.0f})",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=indicadores['superacion_1000_color'],
                    bgcolor="white",
                    bordercolor=indicadores['superacion_1000_color'],
                    borderwidth=2
                )
            else:
                fig.add_annotation(
                    x=año_final,
                    y=poblacion_final,
                    text=f"⚠️ No supera 1000 hab<br>({poblacion_final:,.0f})",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=indicadores['superacion_1000_color'],
                    bgcolor="white",
                    bordercolor=indicadores['superacion_1000_color'],
                    borderwidth=2
                )
            
            fig.update_layout(
                title=f'Proyección Demográfica - {territorio} (Modelo {modelo.title()})',
                xaxis_title='Año',
                yaxis_title='Población Total',
                hovermode='x unified',
                height=500
            )
            
            graficos['principal'] = fig
        
        return graficos
    
    def calcular_indicadores_derivados(self, resultado: Dict) -> Dict:
        """
        Calcula indicadores derivados de las proyecciones
        """
        proyecciones = resultado['proyecciones']
        territorio = resultado['territorio']
        
        indicadores = {}
        
        if resultado['modelo'] == 'comparar_todos':
            # Calcular indicadores para cada modelo
            for modelo, datos_modelo in proyecciones.items():
                indicadores[modelo] = self._calcular_indicadores_modelo(datos_modelo, territorio)
        else:
            # Calcular indicadores para un solo modelo
            indicadores = self._calcular_indicadores_modelo(proyecciones, territorio)
        
        return indicadores
    
    def _calcular_indicadores_modelo(self, proyecciones: Dict, territorio: str) -> Dict:
        """
        Calcula indicadores para un modelo específico incluyendo indicadores socioeconómicos
        """
        indicadores = {}
        
        # Obtener datos del primer y último año
        primer_año = min(proyecciones.keys())
        ultimo_año = max(proyecciones.keys())
        
        poblacion_inicial = proyecciones[primer_año]['poblacion_total']
        poblacion_final = proyecciones[ultimo_año]['poblacion_total']
        
        # Calcular indicadores básicos
        indicadores['crecimiento_total'] = poblacion_final - poblacion_inicial
        indicadores['tasa_crecimiento_total'] = ((poblacion_final / poblacion_inicial) - 1) * 100
        indicadores['tasa_crecimiento_anual_promedio'] = indicadores['tasa_crecimiento_total'] / (ultimo_año - primer_año)
        
        # Calcular riesgo de despoblación (criterio específico)
        if indicadores['tasa_crecimiento_anual_promedio'] < -1:
            indicadores['riesgo_despoblacion'] = 'Alto'
            indicadores['riesgo_despoblacion_color'] = '#d73027'  # Rojo
        elif indicadores['tasa_crecimiento_anual_promedio'] < 0:
            indicadores['riesgo_despoblacion'] = 'Medio'
            indicadores['riesgo_despoblacion_color'] = '#fc8d59'  # Naranja
        else:
            indicadores['riesgo_despoblacion'] = 'Bajo'
            indicadores['riesgo_despoblacion_color'] = '#91cf60'  # Verde
        
        # Verificar si puede superar 1000 habitantes
        poblacion_maxima = max([proyecciones[año]['poblacion_total'] for año in proyecciones.keys()])
        indicadores['puede_superar_1000'] = poblacion_maxima > 1000
        indicadores['poblacion_maxima'] = poblacion_maxima
        
        # Color para visualización de superación de 1000 habitantes
        if indicadores['puede_superar_1000']:
            indicadores['superacion_1000_color'] = '#2ca02c'  # Verde oscuro
        else:
            indicadores['superacion_1000_color'] = '#d62728'  # Rojo oscuro
        
        # Si tenemos datos por componentes, calcular indicadores adicionales
        if 'indice_dependencia_global' in proyecciones[ultimo_año]:
            indicadores['indice_dependencia_final'] = proyecciones[ultimo_año]['indice_dependencia_global']
            indicadores['indice_envejecimiento'] = proyecciones[ultimo_año]['indice_dependencia_mayores']
        
        # Calcular indicadores socioeconómicos si hay datos de migración
        if 'migracion_neta' in proyecciones[ultimo_año]:
            migracion_neta_final = proyecciones[ultimo_año]['migracion_neta']
            indicadores['migracion_neta_final'] = migracion_neta_final
            indicadores['tipo_migracion'] = 'Inmigración neta' if migracion_neta_final > 0 else 'Emigración neta'
            
            # Calcular impacto de migración en el crecimiento
            crecimiento_vegetativo = proyecciones[ultimo_año].get('crecimiento_vegetativo', 0)
            indicadores['impacto_migracion'] = abs(migracion_neta_final) / abs(crecimiento_vegetativo) if crecimiento_vegetativo != 0 else 0
        
        # Calcular indicadores de sostenibilidad demográfica
        if 'indice_dependencia_global' in proyecciones[ultimo_año]:
            indice_dep = proyecciones[ultimo_año]['indice_dependencia_global']
            if indice_dep > 60:
                indicadores['sostenibilidad_demografica'] = 'Crítica'
                indicadores['sostenibilidad_color'] = '#d73027'
            elif indice_dep > 50:
                indicadores['sostenibilidad_demografica'] = 'Preocupante'
                indicadores['sostenibilidad_color'] = '#fc8d59'
            else:
                indicadores['sostenibilidad_demografica'] = 'Favorable'
                indicadores['sostenibilidad_color'] = '#91cf60'
        
        return indicadores


def ejecutar_proyeccion_demografica(territorio: str, años: int, modelo: str, 
                                  poblacion_actual: float) -> Dict:
    """
    Función principal para ejecutar proyecciones demográficas
    
    Args:
        territorio: Nombre del territorio
        años: Número de años a proyectar
        modelo: Tipo de modelo a usar
        poblacion_actual: Población actual del territorio
        
    Returns:
        Diccionario con resultados completos de la proyección
    """
    # Crear instancia del sistema de proyecciones
    sistema_proyecciones = ProyeccionesDemograficas()
    
    # Ejecutar proyección
    resultado = sistema_proyecciones.proyectar_demografia(
        territorio, poblacion_actual, años, modelo
    )
    
    if not resultado:
        return {}
    
    # Calcular indicadores derivados (antes de generar gráficos)
    indicadores = sistema_proyecciones.calcular_indicadores_derivados(resultado)
    resultado['indicadores'] = indicadores

    # Generar gráficos (requiere indicadores para colorear y anotar)
    graficos = sistema_proyecciones.generar_graficos_proyeccion(resultado)
    resultado['graficos'] = graficos
    
    return resultado


def mostrar_resultados_proyeccion(resultado: Dict):
    """
    Muestra los resultados de la proyección en Streamlit con alertas visuales específicas
    
    Args:
        resultado: Diccionario con resultados de la proyección
    """
    if not resultado:
        st.error("No se pudieron calcular las proyecciones")
        return
    
    territorio = resultado['territorio']
    modelo = resultado['modelo']
    indicadores = resultado['indicadores']
    graficos = resultado['graficos']
    
    # Mostrar alertas visuales específicas
    st.subheader(f"🚨 Alertas Demográficas - {territorio}")
    
    if modelo == 'comparar_todos':
        # Mostrar alertas para todos los modelos
        col1, col2, col3 = st.columns(3)
        
        modelos_info = [
            ('lineal', 'Modelo Lineal'),
            ('exponencial', 'Modelo Exponencial'),
            ('componentes', 'Modelo Componentes')
        ]
        
        for i, (modelo_key, modelo_nombre) in enumerate(modelos_info):
            with [col1, col2, col3][i]:
                ind_modelo = indicadores[modelo_key]
                
                # Alerta de riesgo de despoblación
                riesgo_color = ind_modelo['riesgo_despoblacion_color']
                riesgo_texto = ind_modelo['riesgo_despoblacion']
                
                if riesgo_texto == 'Alto':
                    st.error(f"🔴 **{modelo_nombre}**: Riesgo ALTO de despoblación")
                elif riesgo_texto == 'Medio':
                    st.warning(f"🟡 **{modelo_nombre}**: Riesgo MEDIO de despoblación")
                else:
                    st.success(f"🟢 **{modelo_nombre}**: Riesgo BAJO de despoblación")
                
                # Alerta de superación de 1000 habitantes
                if ind_modelo['puede_superar_1000']:
                    st.success(f"✅ **{modelo_nombre}**: Puede superar 1000 hab (máx: {ind_modelo['poblacion_maxima']:,.0f})")
                else:
                    st.error(f"❌ **{modelo_nombre}**: NO supera 1000 hab (máx: {ind_modelo['poblacion_maxima']:,.0f})")
    else:
        # Mostrar alertas para un solo modelo
        col1, col2 = st.columns(2)
        
        with col1:
            # Alerta de riesgo de despoblación
            riesgo_color = indicadores['riesgo_despoblacion_color']
            riesgo_texto = indicadores['riesgo_despoblacion']
            
            if riesgo_texto == 'Alto':
                st.error("🔴 **RIESGO ALTO DE DESPOBLACIÓN**")
                st.error(f"Tasa de crecimiento anual: {indicadores['tasa_crecimiento_anual_promedio']:.2f}%")
            elif riesgo_texto == 'Medio':
                st.warning("🟡 **RIESGO MEDIO DE DESPOBLACIÓN**")
                st.warning(f"Tasa de crecimiento anual: {indicadores['tasa_crecimiento_anual_promedio']:.2f}%")
            else:
                st.success("🟢 **RIESGO BAJO DE DESPOBLACIÓN**")
                st.success(f"Tasa de crecimiento anual: {indicadores['tasa_crecimiento_anual_promedio']:.2f}%")
        
        with col2:
            # Alerta de superación de 1000 habitantes
            if indicadores['puede_superar_1000']:
                st.success("✅ **PUEDE SUPERAR 1000 HABITANTES**")
                st.success(f"Población máxima proyectada: {indicadores['poblacion_maxima']:,.0f} habitantes")
            else:
                st.error("❌ **NO SUPERA 1000 HABITANTES**")
                st.error(f"Población máxima proyectada: {indicadores['poblacion_maxima']:,.0f} habitantes")
    
    # Mostrar indicadores principales
    st.subheader(f"📊 Indicadores de Proyección - {territorio}")
    
    if modelo == 'comparar_todos':
        # Mostrar comparación de modelos
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modelo Lineal", 
                     f"{indicadores['lineal']['tasa_crecimiento_anual_promedio']:.2f}%",
                     f"Riesgo: {indicadores['lineal']['riesgo_despoblacion']}")
        
        with col2:
            st.metric("Modelo Exponencial", 
                     f"{indicadores['exponencial']['tasa_crecimiento_anual_promedio']:.2f}%",
                     f"Riesgo: {indicadores['exponencial']['riesgo_despoblacion']}")
        
        with col3:
            st.metric("Modelo Componentes", 
                     f"{indicadores['componentes']['tasa_crecimiento_anual_promedio']:.2f}%",
                     f"Riesgo: {indicadores['componentes']['riesgo_despoblacion']}")
    else:
        # Mostrar indicadores de un solo modelo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Crecimiento Total", 
                     f"{indicadores['crecimiento_total']:,.0f} habitantes")
        
        with col2:
            st.metric("Tasa Anual Promedio", 
                     f"{indicadores['tasa_crecimiento_anual_promedio']:.2f}%")
        
        with col3:
            st.metric("Riesgo Despoblación", 
                     indicadores['riesgo_despoblacion'])
        
        with col4:
            st.metric("Supera 1000 hab", 
                     "SÍ" if indicadores['puede_superar_1000'] else "NO")
        
        # Mostrar indicadores socioeconómicos adicionales si están disponibles
        if 'migracion_neta_final' in indicadores or 'sostenibilidad_demografica' in indicadores:
            st.subheader("📊 Indicadores Socioeconómicos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'migracion_neta_final' in indicadores:
                    migracion = indicadores['migracion_neta_final']
                    tipo_mig = indicadores.get('tipo_migracion', 'Sin datos')
                    st.metric("Migración Neta Final", 
                             f"{migracion:,.0f}",
                             tipo_mig)
            
            with col2:
                if 'sostenibilidad_demografica' in indicadores:
                    sostenibilidad = indicadores['sostenibilidad_demografica']
                    color = indicadores.get('sostenibilidad_color', '#000000')
                    st.metric("Sostenibilidad Demográfica", 
                             sostenibilidad)
            
            with col3:
                if 'impacto_migracion' in indicadores:
                    impacto = indicadores['impacto_migracion']
                    st.metric("Impacto Migración", 
                             f"{impacto:.2f}x")
    
    # Mostrar gráficos
    st.subheader("📈 Visualización de Proyecciones")
    
    if 'comparativo' in graficos:
        st.plotly_chart(graficos['comparativo'], use_container_width=True)
    elif 'principal' in graficos:
        st.plotly_chart(graficos['principal'], use_container_width=True)
    
    # Mostrar tabla de datos
    st.subheader("📋 Datos Detallados")
    
    if modelo == 'comparar_todos':
        # Crear tabla comparativa
        datos_tabla = []
        for nombre_modelo, datos_modelo in resultado['proyecciones'].items():
            for año, datos_año in datos_modelo.items():
                datos_tabla.append({
                    'Modelo': nombre_modelo.title(),
                    'Año': datos_año['año'],
                    'Población': f"{datos_año['poblacion_total']:,.0f}",
                    'Crecimiento Vegetativo': f"{datos_año.get('crecimiento_vegetativo', 0):,.0f}"
                })
        
        df_tabla = pd.DataFrame(datos_tabla)
        st.dataframe(df_tabla, use_container_width=True)
    else:
        # Crear tabla simple
        datos_tabla = []
        for año, datos_año in resultado['proyecciones'].items():
            datos_tabla.append({
                'Año': datos_año['año'],
                'Población Total': f"{datos_año['poblacion_total']:,.0f}",
                'Crecimiento Vegetativo': f"{datos_año.get('crecimiento_vegetativo', 0):,.0f}",
                'Tasa Crecimiento': f"{datos_año.get('tasa_crecimiento', 0):.2f}%"
            })
        
        df_tabla = pd.DataFrame(datos_tabla)
        st.dataframe(df_tabla, use_container_width=True)
    
    # Botón de descarga
    csv_buffer = df_tabla.to_csv(index=False, sep=';', encoding='utf-8')
    st.download_button(
        label="📥 Descargar datos de proyección",
        data=csv_buffer,
        file_name=f"proyeccion_{territorio}_{modelo}.csv",
        mime="text/csv"
    )

