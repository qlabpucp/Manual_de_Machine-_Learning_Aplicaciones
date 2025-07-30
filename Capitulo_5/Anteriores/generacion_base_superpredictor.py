import pandas as pd
import numpy as np

def generar_dataset_con_super_predictores(n_records=3000, n_noisy_features=50):
    """
    Crea un dataset balanceado (50/50) con dos variables "super-predictoras"
    y muchas variables ruidosas para demostrar la ventaja de Random Forest sobre Bagging.
    """
    
    # --- 1. Crear los Super-Predictors ---
    # Generamos dos puntuaciones continuas que serán muy predictivas.
    # Un puntaje bajo indica alta vulnerabilidad.
    puntaje_economico = np.random.randn(n_records)
    puntaje_social = np.random.randn(n_records)
    
    # --- 2. Crear el Target basado en los Super-Predictors (con ruido) ---
    # La vulnerabilidad de un hogar es una combinación de ambos puntajes.
    # Un hogar será "Pobre" si su puntaje combinado es bajo.
    score_final = (puntaje_economico * 0.6) + (puntaje_social * 0.4)
    
    # Añadimos ruido para que la relación no sea perfecta y el problema sea desafiante.
    noise = np.random.normal(0, np.std(score_final) * 0.4, n_records) # 40% de ruido
    score_final_noisy = score_final + noise
    
    # El 50% con el puntaje más bajo será clasificado como "Pobre".
    # Esto garantiza un dataset perfectamente balanceado.
    median_score = np.median(score_final_noisy)
    pobreza_monetaria = (score_final_noisy < median_score).astype(int)
    
    # --- 3. Crear las Variables Ruidosas ---
    # Generamos una matriz de datos aleatorios que no tienen relación con el target.
    # Esto simula un entorno de alta dimensionalidad.
    noisy_data = np.random.randn(n_records, n_noisy_features)
    noisy_columns = [f'Variable_Ruido_{i+1}' for i in range(n_noisy_features)]
    df_noisy = pd.DataFrame(noisy_data, columns=noisy_columns)
    
    # --- 4. Ensamblar el DataFrame Final ---
    df_main = pd.DataFrame({
        'Puntaje_Focalizacion_A': puntaje_economico,
        'Puntaje_Focalizacion_B': puntaje_social,
        'PobrezaMonetaria': pobreza_monetaria
    })
    
    # Combinamos los super-predictores, el target y las variables ruidosas.
    df_final = pd.concat([df_main, df_noisy], axis=1)
    
    # Barajamos las filas para asegurar aleatoriedad.
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reordenamos para tener el target al final.
    columnas = [col for col in df_final.columns if col != 'PobrezaMonetaria'] + ['PobrezaMonetaria']
    df_final = df_final[columnas]
    
    return df_final

if __name__ == "__main__":
    # Generar el dataset
    df_super_predictor = generar_dataset_con_super_predictores()
    
    # Guardar en un nuevo archivo CSV
    output_filename = 'pobreza_super_predictores.csv'
    df_super_predictor.to_csv(output_filename, index=False)
    
    print(f"Base de datos '{output_filename}' creada exitosamente.")
    print(f"Total de registros: {len(df_super_predictor)}")
    print(f"Total de características (predictores): {len(df_super_predictor.columns) - 1}")
    
    print("\nVerificación de la distribución de la variable objetivo:")
    print(df_super_predictor['PobrezaMonetaria'].value_counts(normalize=True))
    
    print("\nPrimeras 5 filas del dataset:")
    print(df_super_predictor.head())