import pandas as pd
import unicodedata
import re

# Leer Territorios.csv
territorios = pd.read_csv('fase1/Territorios.csv', sep=';')
territorios_lista = territorios['Territorio'].astype(str).tolist()

# Leer ieca_export_latitud_longuitud.csv
latlong = pd.read_csv('ieca_export_latitud_longuitud.csv', sep=';')
latlong_lista = latlong['Territorio'].astype(str).unique().tolist()

def normaliza(texto):
    texto = str(texto)
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.lower().strip()
    texto = re.sub(r'\s+', ' ', texto)
    return texto

territorios_norm = set(normaliza(t) for t in territorios_lista)
latlong_norm = set(normaliza(t) for t in latlong_lista)

faltan = [t for t in territorios_lista if normaliza(t) not in latlong_norm]

print('Territorios de Territorios.csv que NO están en ieca_export_latitud_longuitud.csv:')
for t in faltan:
    print(t)
