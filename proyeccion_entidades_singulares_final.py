def obtener_entidades_singulares(territorio_principal: str) -> list:
    """
    Obtiene las entidades singulares asociadas a un territorio principal
    
    Args:
        territorio_principal: Nombre del territorio principal
        
    Returns:
        Lista de diccionarios con información de entidades singulares
    """
    entidades_singulares = []
    
    with open('Territorios.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Saltar encabezado
        parts = line.strip().split(';')
        if len(parts) >= 6:
            territorio = parts[0]
            factor = parts[3]
            singular = parts[4]
            
            if territorio == territorio_principal:
                try:
                    factor_num = float(factor)
                    if factor_num != 1.0:  # Es una entidad singular
                        entidades_singulares.append({
                            'territorio_principal': territorio_principal,
                            'entidad_singular': singular,
                            'factor': factor_num,
                            'porcentaje': factor_num * 100
                        })
                except:
                    continue
    
    return entidades_singulares

def proyectar_entidad_singular(territorio_principal: str, entidad_singular: str, 
                              proyeccion_principal: dict, factor: float) -> dict:
    """
    Proyecta una entidad singular basándose en la proyección de su territorio principal
    
    Args:
        territorio_principal: Nombre del territorio principal
        entidad_singular: Nombre de la entidad singular
        proyeccion_principal: Diccionario con proyección del territorio principal
        factor: Factor numérico de la entidad singular
        
    Returns:
        Diccionario con proyección de la entidad singular
    """
    años = len(proyeccion_principal['poblacion_total'])
    
    proyeccion_singular = {
        'territorio_principal': territorio_principal,
        'entidad_singular': entidad_singular,
        'factor': factor,
        'porcentaje': factor * 100,
        'años': list(range(2024, 2024 + años)),
        'poblacion_total': [],
        'poblacion_hombres': [],
        'poblacion_mujeres': [],
        'crecimiento_vegetativo': [],
        'migracion_neta': [],
        'tasa_crecimiento': []
    }
    
    # Aplicar factor a cada año de proyección
    for i in range(años):
        # Población total
        poblacion_total = proyeccion_principal['poblacion_total'][i] * factor
        proyeccion_singular['poblacion_total'].append(poblacion_total)
        
        # Población por sexo (mantener proporción)
        poblacion_hombres = proyeccion_principal['poblacion_hombres'][i] * factor
        poblacion_mujeres = proyeccion_principal['poblacion_mujeres'][i] * factor
        proyeccion_singular['poblacion_hombres'].append(poblacion_hombres)
        proyeccion_singular['poblacion_mujeres'].append(poblacion_mujeres)
        
        # Crecimiento vegetativo
        crecimiento_veg = proyeccion_principal['crecimiento_vegetativo'][i] * factor
        proyeccion_singular['crecimiento_vegetativo'].append(crecimiento_veg)
        
        # Migración neta
        migracion_neta = proyeccion_principal['migracion_neta'][i] * factor
        proyeccion_singular['migracion_neta'].append(migracion_neta)
        
        # Tasa de crecimiento
        if i > 0:
            tasa_crecimiento = ((poblacion_total - proyeccion_singular['poblacion_total'][i-1]) / 
                              proyeccion_singular['poblacion_total'][i-1]) * 100
        else:
            tasa_crecimiento = 0
        proyeccion_singular['tasa_crecimiento'].append(tasa_crecimiento)
    
    return proyeccion_singular

def ejemplo_proyeccion_completa():
    """
    Ejemplo completo de proyección para territorio principal y sus entidades singulares
    """
    # Datos de ejemplo de proyección principal
    territorio_principal = "Andújar"
    años = 5
    
    proyeccion_principal = {
        'poblacion_total': [10000, 10200, 10400, 10600, 10800],
        'poblacion_hombres': [4800, 4896, 4992, 5088, 5184],
        'poblacion_mujeres': [5200, 5304, 5408, 5512, 5616],
        'crecimiento_vegetativo': [50, 52, 54, 56, 58],
        'migracion_neta': [20, 22, 24, 26, 28]
    }
    
    # Obtener entidades singulares
    entidades = obtener_entidades_singulares(territorio_principal)
    
    print(f"=== PROYECCIÓN PARA {territorio_principal.upper()} ===")
    print(f"Entidades singulares encontradas: {len(entidades)}")
    
    for entidad in entidades:
        print(f"\n--- {entidad['entidad_singular']} ({entidad['porcentaje']:.2f}%) ---")
        
        # Proyectar entidad singular
        proyeccion_singular = proyectar_entidad_singular(
            territorio_principal,
            entidad['entidad_singular'],
            proyeccion_principal,
            entidad['factor']
        )
        
        # Mostrar resultados
        for i, año in enumerate(proyeccion_singular['años']):
            poblacion = proyeccion_singular['poblacion_total'][i]
            crecimiento = proyeccion_singular['crecimiento_vegetativo'][i]
            migracion = proyeccion_singular['migracion_neta'][i]
            tasa = proyeccion_singular['tasa_crecimiento'][i]
            
            print(f"{año}: Población: {poblacion:.0f}, Crecimiento: {crecimiento:.1f}, Migración: {migracion:.1f}, Tasa: {tasa:.2f}%")

if __name__ == "__main__":
    ejemplo_proyeccion_completa()
