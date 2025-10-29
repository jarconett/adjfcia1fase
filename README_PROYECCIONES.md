# Motor de Proyecciones Demográficas (Entidades Singulares)

## Descripción

Este motor calcula proyecciones demográficas usando datos históricos de crecimiento vegetativo e índices de dependencia. Es el único motor vigente y soporta municipios y entidades singulares.

## Archivos del Módulo

- `proyeccion_entidades_singulares_final.py`: Motor principal de proyecciones
- `coord.py`: Integra la pestaña de proyecciones en la app

## Estructura de Datos Requerida

### Crecimiento Vegetativo
- Ubicación: `demografia/ieca_export_crec_veg_[provincia].csv` (Granada: `gra1`/`gra2`; Jaén: fallback `jaen`)
- Columnas: `Lugar de residencia`/`Lugar de origen`/`Territorio`/`Municipio`/`Lugar`, `Anual`, `Sexo`, `Medida`, `Valor`

### Índices de Dependencia
- Ubicación: `demografia/ieca_export_dep_[provincia]{1,2}.csv`
- Columnas: `Lugar de residencia`/`Lugar de origen`/`Territorio`/`Municipio`/`Lugar`, `Anual`, `Edad`, `Medida`, `Valor`

## Funcionalidades Implementadas

### 1. Análisis de Tendencias Históricas
- Regresión lineal para crecimiento vegetativo por sexo
- Análisis de tendencias de índices de dependencia
- Detección de puntos de inflexión
- Cálculo de estadísticas descriptivas

### 2. Modelos de Proyección
- Tendencia Lineal, Exponencial y Por Componentes

### 3. Indicadores Calculados
- Población total proyectada
- Tasa de crecimiento anual promedio
- Índices de dependencia proyectados
- Riesgo de despoblación
- Coeficientes de determinación (R²)

### 4. Visualizaciones
- Gráficos interactivos con Plotly (umbral 1000 hab)
- Tablas de datos detalladas y exportación a CSV

## Uso en Streamlit

### Acceso a la Funcionalidad
1. Ir a la pestaña "📈 Proyecciones Demográficas (Entidades singulares)"
2. Seleccionar territorio, horizonte temporal y modelo
3. Presionar "🚀 Calcular Proyección Demográfica"

### Configuración Disponible
- **Territorio**: Lista de territorios disponibles
- **Horizonte temporal**: 5, 10, 15, 20 años
- **Modelo**: Lineal, Exponencial, Componentes

## Indicadores Adicionales Recomendados

Para mejorar la precisión de las proyecciones, se recomienda incluir:

### Datos Demográficos Básicos
- **Población por grupos de edad quinquenales** (0-4, 5-9, ..., 80+)
- **Tasas de natalidad por edad de la madre**
- **Tasas de mortalidad por edad y sexo**
- **Migración neta por grupos de edad**

### Datos Socioeconómicos
- **Renta per cápita**
- **Tasa de desempleo**
- **Nivel educativo**
- **Acceso a servicios básicos**

### Datos de Vivienda
- **Precio medio de vivienda**
- **Tasa de ocupación de viviendas**
- **Nuevas construcciones**

### Datos de Servicios
- **Acceso a servicios sanitarios**
- **Acceso a servicios educativos**
- **Acceso a transporte público**

## Limitaciones Actuales

1. **Datos limitados**: Solo crecimiento vegetativo e índices de dependencia
2. **Simplificaciones**: Estimaciones de población por edad basadas en índices
3. **No considera migración**: Solo crecimiento natural
4. **Eventos imprevistos**: No predice crisis o cambios estructurales

## Mejoras Futuras Sugeridas

### Corto Plazo
- Integrar datos de migración neta
- Mejorar estimación de población por edad
- Añadir intervalos de confianza

### Medio Plazo
- Implementar modelo de cohortes completo
- Integrar datos socioeconómicos
- Añadir proyecciones por escenarios

### Largo Plazo
- Modelo de microsimulación
- Integración con datos de servicios públicos
- Predicción de eventos demográficos

## Dependencias

- `pandas`: Manipulación de datos
- `numpy`: Cálculos numéricos
- `plotly`: Visualizaciones interactivas
- `streamlit`: Interfaz de usuario

## Instalación

El módulo se integra automáticamente en el sistema principal. Solo requiere:

1. Archivos de datos en la carpeta `demografia/`
2. Importación en `coord.py`

## Soporte

Para problemas o mejoras, revisar:
- Estructura de archivos de datos
- Disponibilidad de archivos CSV
- Formato de datos (separador `;`, decimal `,`)
- Codificación UTF-8

## Nuevas Funcionalidades Implementadas

### Filtrado por Territorios con Farmacia
- Solo procesa territorios que tienen farmacia según `Territorios.csv`
- Cruza automáticamente `Lugar de residencia` (archivos demografía) con `Territorio` (Territorios.csv)
- Verifica disponibilidad de datos demográficos antes de mostrar opciones

### Alertas Visuales Específicas
- **Riesgo de Despoblación**: 
  - 🔴 Alto: Tasa crecimiento < -1%
  - 🟡 Medio: Tasa crecimiento entre -1% y 0%
  - 🟢 Bajo: Tasa crecimiento ≥ 0%

- **Superación de 1000 Habitantes**:
  - ✅ Verde: Puede superar 1000 habitantes
  - ❌ Rojo: No supera 1000 habitantes

### Visualizaciones Mejoradas
- Líneas de colores según riesgo de despoblación
- Línea de referencia en 1000 habitantes
- Anotaciones automáticas en gráficos
- Métricas destacadas con colores específicos

### Criterios de Evaluación
- Riesgo de Despoblación: basado en tasa de crecimiento anual promedio
- Umbral 1000 Habitantes: población máxima proyectada en el período
- Indicadores Derivados: cálculo automático de todos los indicadores relevantes
