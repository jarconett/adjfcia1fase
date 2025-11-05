pip install streamlit pandas streamlit_folium folium geopy plotly numpy 

https://www.juntadeandalucia.es/institutodeestadisticaycartografia/badea/informe/anual?CodOper=b3_151&idNode=23204
streamlit run C:\Users\Javier\Desktop\adjfcia1fase\coord.py 


TO-DO:

calculo facturacion segun habitantes y rango de edad
densidad de poblacion 
enlaces idealista compra alquiler
cercania playa

Modelos de proyección demográfica
Resumen de los tres modelos:
1. Modelo Lineal (_proyectar_lineal)
Cómo funciona:
Usa una regresión lineal histórica del crecimiento vegetativo.
Fórmula: crecimiento_proj = pendiente × año + intercepto
Proyecta el crecimiento vegetativo de forma constante año a año.
Aplica una tasa de crecimiento a la población actual.
Características:
Cuándo usar:
Población con tendencia histórica estable (crecimiento o decrecimiento constante).
Escenarios de corto y medio plazo (5-15 años).
Simplicidad.
2. Modelo Exponencial (_proyectar_exponencial)
Cómo funciona:
Usa la tasa de crecimiento promedio histórica.
Fórmula: población_futura = población_actual × (1 + tasa)^años
La tasa es constante; el crecimiento es compuesto (el incremento aumenta con la población base).
Características:
Cuándo usar:
Poblaciones en crecimiento acelerado o declive acelerado.
Escenarios donde el crecimiento se acelera/desacelera con el tiempo.
En periodos largos, los resultados pueden ser más extremos.
3. Modelo por Componentes (_proyectar_componentes)
Cómo funciona:
Desglosa la población por grupos de edad (activa, jóvenes, mayores).
Usa índices de dependencia proyectados (global, jóvenes, mayores).
Calcula componentes demográficos y los suma.
Características:
Cuándo usar:
Análisis detallado del envejecimiento poblacional.
Necesidad de ver la composición por edad.
Planificación de servicios (sanidad, educación, dependencia).
Es más complejo y requiere índices de dependencia históricos.
Comparación resumida
Aspecto	Lineal	Exponencial	Componentes
Complejidad	Baja	Media	Alta
Datos necesarios	Crecimiento vegetativo	Crecimiento vegetativo	Crecimiento + Dependencia
Información adicional	Básica	Básica	Desglose por edad
Útil para	Tendencias estables	Crecimiento acelerado	Análisis demográfico detallado
Horizonte	Corto-medio	Medio-largo	Medio-largo
Envejecimiento	No considera	No considera	Sí considera
Recomendación:
Análisis rápido: Lineal.
Tendencias aceleradas: Exponencial.
Planificación de servicios y envejecimiento: Componentes.
