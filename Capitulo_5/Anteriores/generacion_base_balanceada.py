import pandas as pd
import numpy as np

def generar_base_de_datos_balanceada_desafiante(num_records=3000):
    """
    Genera un DataFrame balanceado (50/50) creando primero una población continua
    y luego asignando la clase de pobreza basada en un índice de vulnerabilidad,
    lo que garantiza un problema de clasificación no trivial.
    """
    
    # 1. GENERAR UNA ÚNICA POBLACIÓN DIVERSA
    data = {
        'EdadJefeHogar': np.random.randint(18, 80, size=num_records),
        'SexoJefeHogar': np.random.choice(['Hombre', 'Mujer'], size=num_records, p=[0.7, 0.3]),
        'MiembrosHogar': np.random.randint(1, 12, size=num_records),
        'RatioDependencia': np.random.uniform(0, 5, size=num_records).round(2),
        'LenguaMaterna': np.random.choice(['Español', 'Quechua', 'Aymara', 'Otra'], size=num_records, p=[0.85, 0.12, 0.02, 0.01]),
        'NivelEducativoJefeHogar': np.random.choice(['Ninguno', 'Primaria', 'Secundaria', 'Superior'], size=num_records, p=[0.15, 0.35, 0.4, 0.1]),
        'AniosEstudioJefeHogar': np.random.randint(0, 22, size=num_records),
        'TipoEmpleo': np.random.choice(['Formal', 'Informal', 'Desempleado'], size=num_records, p=[0.4, 0.5, 0.1]),
        'IngresoMensualHogar': np.random.lognormal(mean=7.5, sigma=0.8, size=num_records).round(2) * 3,
        'AreaResidencia': np.random.choice(['Urbana', 'Rural'], size=num_records, p=[0.7, 0.3]),
        'TenenciaVivienda': np.random.choice(['Propia', 'Alquilada', 'Cedida'], size=num_records, p=[0.6, 0.3, 0.1]),
        'MaterialParedes': np.random.choice(['Ladrillo/Cemento', 'Adobe', 'Madera/Esteras'], size=num_records, p=[0.6, 0.25, 0.15]),
        'AccesoAguaPotable': np.random.choice([1, 0], size=num_records, p=[0.8, 0.2]),
        'AccesoSaneamiento': np.random.choice([1, 0], size=num_records, p=[0.75, 0.25]),
        'AccesoElectricidad': np.random.choice([1, 0], size=num_records, p=[0.9, 0.1]),
        'Hacinamiento': np.random.choice([1, 0], size=num_records, p=[0.4, 0.6]),
        'PoseeActivos': np.random.choice([1, 0], size=num_records, p=[0.5, 0.5]),
        'Region': np.random.choice(['Costa', 'Sierra', 'Selva'], size=num_records, p=[0.5, 0.3, 0.2])
    }
    df = pd.DataFrame(data)
    df['GastoMensualHogar'] = (df['IngresoMensualHogar'] * np.random.uniform(0.7, 1.2, size=num_records)).round(2)
    
    # 2. CALCULAR UN ÍNDICE DE VULNERABILIDAD (PUNTUACIÓN DE POBREZA)
    # Un puntaje más alto significa más vulnerable.
    vulnerability_score = (
        - (df['IngresoMensualHogar'] / 1000) * 1.5
        - df['AniosEstudioJefeHogar'] * 0.3
        + df['MiembrosHogar'] * 0.5
        + df['RatioDependencia'] * 0.4
        - (df['PoseeActivos'] * 1.0)
        + (df['Hacinamiento'] * 1.2)
        + (df['AreaResidencia'] == 'Rural') * 2.0
        + (df['TipoEmpleo'] != 'Formal') * 1.5
        - (df['NivelEducativoJefeHogar'] == 'Superior') * 2.0
    )
    
    # Añadir ruido aleatorio para que la relación no sea perfecta
    noise = np.random.normal(0, np.std(vulnerability_score) * 0.5, num_records) # 50% de ruido
    df['vulnerability_score'] = vulnerability_score + noise

    # 3. DEFINIR LA POBREZA BASADA EN EL RANKING
    # El 50% con el puntaje más alto será clasificado como "Pobre"
    threshold = df['vulnerability_score'].median()
    df['PobrezaMonetaria'] = (df['vulnerability_score'] >= threshold).astype(int)
    
    # Limpieza final: eliminamos el índice de vulnerabilidad para que no sea usado como predictor
    df = df.drop(columns=['vulnerability_score'])
    
    # Reordenar columnas
    columnas = [col for col in df.columns if col != 'PobrezaMonetaria'] + ['PobrezaMonetaria']
    df = df[columnas]
    
    return df

if __name__ == "__main__":
    df_balanceado_final = generar_base_de_datos_balanceada_desafiante(num_records=3000)
    
    output_filename = 'prediccion_pobreza_peru_balanceada.csv'
    df_balanceado_final.to_csv(output_filename, index=False)

    print(f"Base de datos '{output_filename}' creada exitosamente con un problema desafiante.")
    print(f"Total de registros: {len(df_balanceado_final)}")
    
    print("\nDistribución de la variable objetivo (debe ser 50/50):")
    print(df_balanceado_final['PobrezaMonetaria'].value_counts(normalize=True))