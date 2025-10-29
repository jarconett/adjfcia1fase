pip install streamlit pandas streamlit_folium folium geopy plotly numpy 


streamlit run C:\Users\Javier\Desktop\adjfcia1fase\coord.py 


TO-DO:
arreglar coordenadas de las entidades singulares.
calculo facturacion segun habitantes y rango de edad
densidad de poblacion 
enlaces idealista compra alquiler
cercania playa

factores de correlación con renta o archivo det




N;Provincia;Territorio
1;Almería;Albox
2;Almería;Aldeire
3;Almería;Albondón
4;Almería;Alquife
5;Almería;Lúcar
6;Almería;Lucainena de las Torres
7;Almería;Turrillas
8;Almería;Paterna del Río
9;Almería;Terque
10;Almería;Gorafe
11;Almería;Vícar
12;Almería;Velefique
13;Almería;Tahal
14;Almería;Ragol
15;Almería;Chercos
16;Almería;Cóbdar
17;Almería;Sierro
18;Almería;Sufli
19;Almería;Padules
20;Almería;Gor
21;Almería;Ohanes
22;Almería;Lanteira
23;Almería;Bayarque
24;Almería;Sorvilán
25;Almería;Instinción
26;Almería;Alcontar
27;Almería;Cogollos de Guadix
28;Almería;Enix
29;Almería;Ferreira
30;Almería;Albánchez
31;Almería;La Calahorra
32;Almería;Alhabía
33;Almería;Turón
34;Almería;Almegíjar
35;Almería;Villanueva de las Torres
36;Almería;Lugros
37;Almería;Bentarique
38;Almería;Almócita
1;Cádiz;Villaluenga del Rosario
2;Cádiz;Grazalema
3;Cádiz;Jerez de la Frontera
4;Cádiz;Medina Sidonia
5;Cádiz;Tarifa
6;Cádiz;Torre Alháquime
7;Cádiz;El Puerto de Santa María
1;Córdoba;El Guijo
2;Córdoba;Los Blázquez
3;Córdoba;Adamuz
4;Córdoba;Obejo
5;Córdoba;Córdoba
6;Córdoba;Alcolea
7;Córdoba;Castro del Río
8;Córdoba;Conquista
1;Granada;Domingo Pérez de Granada
2;Granada;Loja
3;Granada;El Valle
4;Granada;Trevélez
5;Granada;Pórtugos
6;Granada;Lentegí
7;Granada;Salobreña
8;Granada;Alpujarra de la Sierra
1;Huelva;Sanlúcar de Guadiana
2;Huelva;Cabezas Rubias
3;Huelva;Berrocal
4;Huelva;Puerto Moral
5;Huelva;Arroyomolinos de León
6;Huelva;Castaño del Robledo
7;Huelva;Cañaveral de León
8;Huelva;Hinojales
9;Huelva;Valdelarco
10;Huelva;La Granada de Río-tinto
1;Jaén;Andújar
2;Jaén;Aldeaquemada
3;Jaén;Higuera de Calatrava
4;Jaén;Jaén
5;Jaén;Martos
6;Jaén;Castillo de Locubín
7;Jaén;La Iruela
8;Jaén;Cazalilla
9;Jaén;Santiago-Pontones
10;Jaén;Villarrodrigo
11;Jaén;Benatae
1;Málaga;Canillas de Albaida
2;Málaga;Cútar
3;Málaga;Pujerra
4;Málaga;Vélez-Málaga
5;Málaga;Alfarnatejo
6;Málaga;Nerja
7;Málaga;Faraján
8;Málaga;Alhaurín el Grande
9;Málaga;Iznájar
10;Málaga;Serrato
11;Málaga;Salares
1;Sevilla;Lora del Río
2;Sevilla;Alcalá del Río
3;Sevilla;Villanueva del Río y Minas
4;Sevilla;La Rinconada
5;Sevilla;Castilleja del Campo

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
