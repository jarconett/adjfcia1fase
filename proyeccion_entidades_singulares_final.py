"""
Módulo de Proyección para Entidades Singulares y Municipios
Independiente del resto de módulos. No reutiliza funciones de coord.py

Lee datos de la carpeta `demografia/` y de `Territorios.csv` y `singular_pob_sexo.csv`.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Utilidades
# -----------------------------

def _normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower().strip()


_TERR_COL_CANDIDATAS: List[str] = [
    "Lugar de residencia",
    "Lugar de origen",
    "Lugar de procedencia",
    "Territorio",
    "Municipio",
    "Lugar",
]


def _columna_territorio(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalizadas = { _normalizar_texto(c): c for c in df.columns }
    for cand in _TERR_COL_CANDIDATAS:
        c_norm = _normalizar_texto(cand)
        if c_norm in normalizadas:
            return normalizadas[c_norm]
    return None


def _cargar_territorios_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv("Territorios.csv", sep=";")
        # Limpieza básica
        for col in ["Territorio", "Provincia", "Singular"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _mapa_provincia_a_codigo() -> Dict[str, str]:
    # Códigos de ficheros
    return {
        "almeria": "alm",
        "cadiz": "cad",
        "cordoba": "cor",
        "granada": "gra",
        "huelva": "hue",
        "jaen": "jae",  # dep usa 'jae'; crecimiento tiene 'jaen' (se maneja más abajo)
        "malaga": "mal",
        "sevilla": "sev",
    }


def _obtener_codigo_provincia_por_territorio(territorio: str) -> Optional[str]:
    df = _cargar_territorios_csv()
    if df.empty:
        return None
    mapa = _mapa_provincia_a_codigo()
    t_norm = _normalizar_texto(territorio)
    # Buscar fila por Territorio exacto
    if "Territorio" in df.columns:
        df["__t_norm"] = df["Territorio"].astype(str).map(_normalizar_texto)
        filas = df[df["__t_norm"] == t_norm]
        if not filas.empty:
            prov = _normalizar_texto(str(filas.iloc[0].get("Provincia", "")))
            return mapa.get(prov)
    # Búsqueda por coincidencia parcial
    for _, row in df.iterrows():
        prov = _normalizar_texto(str(row.get("Provincia", "")))
        terr = _normalizar_texto(str(row.get("Territorio", "")))
        if t_norm in terr or terr in t_norm:
            if prov in _mapa_provincia_a_codigo():
                return _mapa_provincia_a_codigo()[prov]
    return None


# -----------------------------
# Carga de datos IECA
# -----------------------------

def _cargar_crecimiento_vegetativo(territorio: str) -> pd.DataFrame:
    """Carga datos de crecimiento vegetativo para el municipio del territorio."""
    codigo = _obtener_codigo_provincia_por_territorio(territorio)
    if not codigo:
        return pd.DataFrame()

    # Granada tiene doble fichero
    try:
        if codigo == "gra":
            df1 = pd.read_csv("demografia/ieca_export_crec_veg_gra1.csv", sep=";", decimal=",")
            df2 = pd.read_csv("demografia/ieca_export_crec_veg_gra2.csv", sep=";", decimal=",")
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            ruta = f"demografia/ieca_export_crec_veg_{codigo}.csv"
            try:
                df = pd.read_csv(ruta, sep=";", decimal=",")
            except FileNotFoundError:
                # Jaén: algunos datasets pueden venir como 'jaen'
                if codigo == "jae":
                    df = pd.read_csv("demografia/ieca_export_crec_veg_jaen.csv", sep=";", decimal=",")
                else:
                    raise
    except Exception:
        return pd.DataFrame()

    col_terr = _columna_territorio(df)
    if not col_terr:
        return pd.DataFrame()

    df = df.copy()
    df["__t_norm"] = df[col_terr].astype(str).map(_normalizar_texto)
    t_norm = _normalizar_texto(territorio)
    df = df[df["__t_norm"] == t_norm].drop(columns=["__t_norm"], errors="ignore").copy()

    # Asegurar tipos
    if "Anual" in df.columns:
        df["Anual"] = pd.to_numeric(df["Anual"], errors="coerce")
    if "Valor" in df.columns:
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df = df.dropna(subset=["Anual", "Valor"]).copy()
    return df


def _cargar_dependencia(territorio: str) -> pd.DataFrame:
    """Carga índices de dependencia para el municipio del territorio."""
    codigo = _obtener_codigo_provincia_por_territorio(territorio)
    if not codigo:
        return pd.DataFrame()

    try:
        df1 = pd.read_csv(f"demografia/ieca_export_dep_{codigo}1.csv", sep=";", decimal=",")
    except FileNotFoundError:
        if codigo == "jae":
            df1 = pd.read_csv("demografia/ieca_export_dep_jae1.csv", sep=";", decimal=",")
        else:
            return pd.DataFrame()
    try:
        df2 = pd.read_csv(f"demografia/ieca_export_dep_{codigo}2.csv", sep=";", decimal=",")
    except FileNotFoundError:
        if codigo == "jae":
            df2 = pd.read_csv("demografia/ieca_export_dep_jae2.csv", sep=";", decimal=",")
        else:
            return pd.DataFrame()

    df = pd.concat([df1, df2], ignore_index=True)
    col_terr = _columna_territorio(df)
    if not col_terr:
        return pd.DataFrame()

    df = df.copy()
    df["__t_norm"] = df[col_terr].astype(str).map(_normalizar_texto)
    t_norm = _normalizar_texto(territorio)
    df = df[df["__t_norm"] == t_norm].drop(columns=["__t_norm"], errors="ignore").copy()

    if "Anual" in df.columns:
        df["Anual"] = pd.to_numeric(df["Anual"], errors="coerce")
    if "Valor" in df.columns:
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df = df.dropna(subset=["Anual", "Valor"]).copy()
    return df


# -----------------------------
# Población actual
# -----------------------------

def obtener_poblacion_actual(territorio: str, ambito: str = "municipio", singular: Optional[str] = None) -> Optional[float]:
    """Obtiene población 'Ambos sexos' desde singular_pob_sexo.csv.
    - ambito = 'municipio' -> usa nombre de municipio
    - ambito = 'singular' -> usa cadena exacta de la columna 'Singular' en Territorios.csv
    """
    try:
        df = pd.read_csv("singular_pob_sexo.csv", sep=";")
    except Exception:
        return None

    if ambito == "singular" and singular:
        clave = str(singular).strip()
    else:
        clave = str(territorio).strip()

    df_ok = df[(df["Territorio"] == clave) & (df["Sexo"] == "Ambos sexos") & (df["Medida"] == "Población")]
    if df_ok.empty:
        return None
    val = df_ok.iloc[0]["Valor"]
    try:
        return float(val)
    except Exception:
        return None


def _obtener_factor_singular(municipio: str, singular: str) -> Optional[float]:
    """Obtiene Factor desde Territorios.csv para el par (municipio, singular)."""
    df = _cargar_territorios_csv()
    if df.empty:
        return None
    df = df.copy()
    df["__terr_norm"] = df.get("Territorio", "").astype(str).map(_normalizar_texto)
    df["__sing_norm"] = df.get("Singular", "").astype(str).map(_normalizar_texto)
    t_norm = _normalizar_texto(municipio)
    s_norm = _normalizar_texto(singular)
    filas = df[(df["__terr_norm"] == t_norm) & (df["__sing_norm"] == s_norm)]
    if filas.empty:
        return None
    try:
        return float(pd.to_numeric(filas.iloc[0].get("Factor", 1.0), errors="coerce"))
    except Exception:
        return None


def _obtener_poblacion_municipio_normalizada(municipio: str) -> Optional[float]:
    """Busca población del municipio normalizando nombres y probando órdenes de columnas."""
    try:
        df = pd.read_csv("singular_pob_sexo.csv", sep=";")
    except Exception:
        return None
    df = df.copy()
    df["__terr_norm"] = df.get("Territorio", "").astype(str).map(_normalizar_texto)
    t_norm = _normalizar_texto(municipio)
    # estándar
    m1 = df[(df["__terr_norm"] == t_norm) & (df.get("Sexo", "") == "Ambos sexos") & (df.get("Medida", "") == "Población")]
    # alterno
    m2 = df[(df["__terr_norm"] == t_norm) & (df.get("Medida", "") == "Ambos sexos") & (df.get("Sexo", "") == "Población")]
    m = m1 if not m1.empty else m2
    if m.empty:
        return None
    try:
        return float(m.iloc[0]["Valor"])
    except Exception:
        return None


# -----------------------------
# Tendencias y Proyecciones
# -----------------------------

def _tendencias_crecimiento(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    tendencias: Dict[str, Dict[str, float]] = {}
    for sexo in ["Ambos sexos", "Hombres", "Mujeres"]:
        datos = df[df.get("Sexo", "") == sexo].copy()
        if datos.empty:
            continue
        x = datos["Anual"].values
        y = datos["Valor"].values
        try:
            pendiente, intercepto = np.polyfit(x, y, 1)
        except Exception:
            continue
        y_pred = pendiente * x + intercepto
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2) if np.sum((y - np.mean(y)) ** 2) != 0 else 1)
        tasa_media = float(np.mean(np.diff(y) / y[:-1]) * 100) if len(y) > 1 and np.all(y[:-1] != 0) else 0.0
        tendencias[_normalizar_texto(sexo).replace(" ", "_")] = {
            "pendiente": float(pendiente),
            "intercepto": float(intercepto),
            "r_squared": float(r2),
            "tasa_crecimiento_promedio": float(tasa_media),
            "valor_ultimo": float(y[-1]),
            "año_ultimo": int(x[-1]),
            "valor_primer": float(y[0]),
            "año_primer": int(x[0]),
        }
    return tendencias


def _tendencias_dependencia(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    tendencias: Dict[str, Dict[str, float]] = {}
    alias = {
        "índice de dependencia global": "global",
        "índice de dependencia jóvenes": "jovenes",
        "índice de dependencia mayores": "mayores",
    }
    if "Edad" not in df.columns:
        return tendencias
    for etiqueta, clave in alias.items():
        datos = df[df["Edad"].str.lower() == etiqueta].copy()
        if datos.empty:
            continue
        x = datos["Anual"].values
        y = datos["Valor"].values
        try:
            pendiente, intercepto = np.polyfit(x, y, 1)
        except Exception:
            continue
        y_pred = pendiente * x + intercepto
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2) if np.sum((y - np.mean(y)) ** 2) != 0 else 1)
        tendencias[clave] = {
            "pendiente": float(pendiente),
            "intercepto": float(intercepto),
            "r_squared": float(r2),
            "cambio_anual_promedio": float(pendiente),
            "valor_ultimo": float(y[-1]),
            "año_ultimo": int(x[-1]),
            "valor_primer": float(y[0]),
            "año_primer": int(x[0]),
        }
    return tendencias


def _proyectar_lineal(poblacion_actual: float, tendencias: Dict, años: int) -> Dict[int, Dict[str, float]]:
    crec = tendencias.get("crecimiento", {}).get("ambos_sexos")
    if not crec:
        return {}
    m = crec["pendiente"]
    b = crec["intercepto"]
    año_base = crec["año_ultimo"]
    out: Dict[int, Dict[str, float]] = {}
    for i in range(1, años + 1):
        a = año_base + i
        crecimiento_proj = m * a + b
        tasa = (crecimiento_proj / poblacion_actual) if poblacion_actual else 0.0
        pobl = poblacion_actual * (1 + tasa) ** i if poblacion_actual else 0.0
        out[i] = {
            "año": a,
            "poblacion_total": float(pobl),
            "crecimiento_vegetativo": float(crecimiento_proj),
            "tasa_crecimiento": float(tasa * 100),
        }
    return out


def _proyectar_exponencial(poblacion_actual: float, tendencias: Dict, años: int) -> Dict[int, Dict[str, float]]:
    crec = tendencias.get("crecimiento", {}).get("ambos_sexos")
    if not crec:
        return {}
    tasa = (crec.get("tasa_crecimiento_promedio", 0.0) or 0.0) / 100.0
    año_base = crec.get("año_ultimo", 0)
    out: Dict[int, Dict[str, float]] = {}
    for i in range(1, años + 1):
        a = año_base + i
        pobl = poblacion_actual * (1 + tasa) ** i if poblacion_actual else 0.0
        crec_proj = pobl * tasa
        out[i] = {
            "año": a,
            "poblacion_total": float(pobl),
            "crecimiento_vegetativo": float(crec_proj),
            "tasa_crecimiento": float(tasa * 100),
        }
    return out


def _proyectar_componentes(poblacion_actual: float, tendencias: Dict, años: int) -> Dict[int, Dict[str, float]]:
    dep_glob = tendencias.get("dependencia", {}).get("global")
    dep_may = tendencias.get("dependencia", {}).get("mayores")
    if not dep_glob or not dep_may:
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for i in range(1, años + 1):
        a = dep_glob["año_ultimo"] + i
        idx_g = dep_glob["pendiente"] * a + dep_glob["intercepto"]
        idx_m = dep_may["pendiente"] * a + dep_may["intercepto"]
        pob_activa = poblacion_actual * 0.65 if poblacion_actual else 0.0
        pob_dep = pob_activa * (idx_g / 100.0)
        pob_may = pob_activa * (idx_m / 100.0)
        pob_jov = max(pob_dep - pob_may, 0.0)
        pob_total = pob_activa + pob_dep
        out[i] = {
            "año": a,
            "poblacion_total": float(pob_total),
            "poblacion_activa": float(pob_activa),
            "poblacion_jovenes": float(pob_jov),
            "poblacion_mayores": float(pob_may),
            "indice_dependencia_global": float(idx_g),
            "indice_dependencia_mayores": float(idx_m),
            "indice_dependencia_jovenes": float((pob_jov / pob_activa) * 100 if pob_activa else 0.0),
        }
    return out


def _calcular_indicadores(proy: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    ind: Dict[str, float] = {}
    if not proy:
        return ind
    años = sorted(proy.keys())
    ini, fin = años[0], años[-1]
    p_ini = proy[ini]["poblacion_total"]
    p_fin = proy[fin]["poblacion_total"]
    ind["crecimiento_total"] = float(p_fin - p_ini)
    ind["tasa_crecimiento_total"] = float(((p_fin / p_ini) - 1) * 100) if p_ini else 0.0
    ind["tasa_crecimiento_anual_promedio"] = float(ind["tasa_crecimiento_total"] / (fin - ini)) if fin != ini else 0.0
    # Riesgo
    if ind["tasa_crecimiento_anual_promedio"] < -1:
        ind["riesgo_despoblacion"] = "Alto"
        ind["riesgo_despoblacion_color"] = "#d73027"
    elif ind["tasa_crecimiento_anual_promedio"] < 0:
        ind["riesgo_despoblacion"] = "Medio"
        ind["riesgo_despoblacion_color"] = "#fc8d59"
    else:
        ind["riesgo_despoblacion"] = "Bajo"
        ind["riesgo_despoblacion_color"] = "#91cf60"
    p_max = max(proy[a]["poblacion_total"] for a in años)
    ind["puede_superar_1000"] = bool(p_max > 1000)
    ind["poblacion_maxima"] = float(p_max)
    ind["superacion_1000_color"] = "#2ca02c" if ind["puede_superar_1000"] else "#d62728"
    # Extras si existen
    if "indice_dependencia_global" in proy[fin]:
        ind["indice_dependencia_final"] = float(proy[fin]["indice_dependencia_global"])
        ind["indice_envejecimiento"] = float(proy[fin]["indice_dependencia_mayores"])
    return ind


def _graficos(territorio: str, modelo: str, proy: Dict[int, Dict[str, float]], ind: Dict[str, float]) -> Dict[str, go.Figure]:
    graficos: Dict[str, go.Figure] = {}
    años = [proy[k]["año"] for k in sorted(proy.keys())]
    pobl = [proy[k]["poblacion_total"] for k in sorted(proy.keys())]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=años, y=pobl, mode="lines+markers",
        name=f"Proyección {modelo}",
        line=dict(width=4, color=ind.get("riesgo_despoblacion_color", "#1f77b4")),
        marker=dict(size=8)
    ))
    fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="Umbral 1000 hab", annotation_position="top right")
    fig.update_layout(title=f"Proyección Demográfica - {territorio} (Modelo {modelo})", xaxis_title="Año", yaxis_title="Población Total", hovermode="x unified", height=500)
    graficos["principal"] = fig
    return graficos


# -----------------------------
# API pública del módulo
# -----------------------------

def ejecutar_proyeccion_entidades_singulares(
    territorio_municipal: str,
    años: int,
    modelo: str,
    poblacion_actual: Optional[float],
) -> Dict:
    """Ejecuta la proyección para un municipio (o entidad singular pasando su población).
    - territorio_municipal: municipio base para tendencias
    - poblacion_actual: población del ámbito elegido (municipio o singular)
    """
    df_crec = _cargar_crecimiento_vegetativo(territorio_municipal)
    df_dep = _cargar_dependencia(territorio_municipal)
    if df_crec.empty or df_dep.empty:
        return {}
    tendencias = {
        "crecimiento": _tendencias_crecimiento(df_crec),
        "dependencia": _tendencias_dependencia(df_dep),
    }
    if not poblacion_actual or poblacion_actual <= 0:
        # fallback: aproximar con último valor positivo si existiera
        pob_fallback = max(float(tendencias["crecimiento"].get("ambos_sexos", {}).get("valor_ultimo", 0.0)), 1.0)
        poblacion_actual = pob_fallback

    if modelo == "lineal":
        proy = _proyectar_lineal(poblacion_actual, tendencias, años)
    elif modelo == "exponencial":
        proy = _proyectar_exponencial(poblacion_actual, tendencias, años)
    elif modelo == "componentes":
        proy = _proyectar_componentes(poblacion_actual, tendencias, años)
    else:
        proy = _proyectar_lineal(poblacion_actual, tendencias, años)

    indicadores = _calcular_indicadores(proy)
    return {
        "territorio": territorio_municipal,
        "modelo": modelo,
        "años_proyeccion": años,
        "proyecciones": proy,
        "tendencias": tendencias,
        "indicadores": indicadores,
        "graficos": _graficos(territorio_municipal, modelo, proy, indicadores),
    }


def cargar_catalogos_entidades() -> Tuple[List[str], List[Tuple[str, str]]]:
    """Devuelve:
    - lista de municipios (Territorio)
    - lista de pares (municipio, singular) para entidades singulares (no vacías)
    """
    df = _cargar_territorios_csv()
    if df.empty:
        return [], []
    municipios = sorted(df["Territorio"].dropna().unique().tolist()) if "Territorio" in df.columns else []
    sing: List[Tuple[str, str]] = []
    if "Singular" in df.columns:
        tmp = df.dropna(subset=["Singular"]).copy()
        for _, r in tmp.iterrows():
            terr = str(r.get("Territorio", "")).strip()
            s = str(r.get("Singular", "")).strip()
            if s:
                sing.append((terr, s))
    return municipios, sing


def render_proyeccion_entidades_singulares():
    """UI independiente para Streamlit en la pestaña de Proyecciones Demográficas."""
    st.subheader("Configuración de Proyección (Motor: Entidades Singulares)")
    municipios, singpares = cargar_catalogos_entidades()
    if not municipios:
        st.error("No se pudieron cargar municipios desde Territorios.csv")
        return

    ambito = st.radio("Ámbito de proyección", ["Municipio", "Entidad singular"], index=0, horizontal=True)
    if ambito == "Municipio":
        municipio = st.selectbox("Municipio", options=municipios)
        # Usar búsqueda robusta (normalización y doble orden de columnas)
        pobl_actual = _obtener_poblacion_municipio_normalizada(municipio)
        st.info(
            f"Población actual (Ambos sexos) de {municipio}: {pobl_actual:,.0f}" if pobl_actual else f"No se pudo obtener población de {municipio}"
        )
        años = st.selectbox("Horizonte (años)", [5, 10, 15, 20], index=1)
        modelo = st.selectbox("Modelo", ["lineal", "exponencial", "componentes"], index=0)
        if st.button("Calcular proyección", use_container_width=True):
            with st.spinner("Calculando proyección..."):
                res = ejecutar_proyeccion_entidades_singulares(municipio, años, modelo, pobl_actual)
            if not res:
                st.error("No se pudo calcular la proyección para el municipio seleccionado")
                return
            _render_resultado(res)
    else:
        if not singpares:
            st.warning("No hay entidades singulares en Territorios.csv")
            return
        opciones = [f"{s} — [{m}]" for (m, s) in singpares]
        sel = st.selectbox("Entidad singular", options=opciones)
        idx = opciones.index(sel)
        municipio, singular = singpares[idx]
        # Metodología: usar población del Territorio (municipio) multiplicada por Factor de Territorios.csv
        pob_muni = _obtener_poblacion_municipio_normalizada(municipio) or 0.0
        factor = _obtener_factor_singular(municipio, singular) or 1.0
        pobl_actual = float(pob_muni) * float(factor)
        if pobl_actual > 0:
            st.info(f"Población estimada de {singular} = Población {municipio} × Factor ({factor:.4f}) → {pobl_actual:,.0f}")
        else:
            st.warning(f"No se pudo obtener la población del municipio {municipio} o factor para {singular}")
        años = st.selectbox("Horizonte (años)", [5, 10, 15, 20], index=1, key="años_sing")
        modelo = st.selectbox("Modelo", ["lineal", "exponencial", "componentes"], index=0, key="modelo_sing")
        if st.button("Calcular proyección para entidad singular", use_container_width=True):
            with st.spinner("Calculando proyección..."):
                # Tendencias por municipio, población del singular
                res = ejecutar_proyeccion_entidades_singulares(municipio, años, modelo, pobl_actual)
                # Cambiar etiqueta a singular para presentación
                if res:
                    res = {**res, "territorio": singular}
            if not res:
                st.error("No se pudo calcular la proyección para la entidad singular seleccionada")
                return
            _render_resultado(res)


def _render_resultado(res: Dict):
    ind = res.get("indicadores", {})
    gra = res.get("graficos", {})
    territorio = res.get("territorio", "?")
    st.subheader(f"📊 Indicadores - {territorio}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Crecimiento Total", f"{ind.get('crecimiento_total', 0):,.0f} hab")
    with c2:
        st.metric("Tasa Anual Promedio", f"{ind.get('tasa_crecimiento_anual_promedio', 0):.2f}%")
    with c3:
        st.metric("Riesgo Despoblación", ind.get("riesgo_despoblacion", "N/A"))
    with c4:
        st.metric("Supera 1000 hab", "SÍ" if ind.get("puede_superar_1000", False) else "NO")

    st.subheader("📈 Proyección")
    if "principal" in gra:
        st.plotly_chart(gra["principal"], use_container_width=True)

    # Tabla
    st.subheader("📋 Datos de proyección")
    filas = []
    for k in sorted(res["proyecciones"].keys()):
        d = res["proyecciones"][k]
        filas.append({
            "Año": d.get("año"),
            "Población Total": f"{d.get('poblacion_total', 0):,.0f}",
            "Crec. Vegetativo": f"{d.get('crecimiento_vegetativo', 0):,.0f}",
            "Tasa Crec.%": f"{d.get('tasa_crecimiento', 0):.2f}%",
        })
    if filas:
        st.dataframe(pd.DataFrame(filas), use_container_width=True)
