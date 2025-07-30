import nbformat as nbf

def generar_notebook_final():
    """
    Crea un Jupyter Notebook completo con todos los pasos para el modelo Random Forest,
    desde la carga de datos hasta la interpretación del modelo.
    """
    # Crear un nuevo notebook
    nb = nbf.v4.new_notebook()
    
    # Lista para almacenar todas las celdas
    celdas = []

    # ==========================================================================
    # --- PASO 1: DEFINICIÓN DEL PROBLEMA Y PREPARACIÓN INICIAL ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
# Modelo Random Forest: Clasificación de Nivel de Contagios
## Paso 1: Definición del Problema y Preparación Inicial

### 🎯 Objetivo de la Actividad
Establecer las bases del proyecto: cargar librerías, importar datos, realizar una exploración inicial y crear nuestra variable objetivo para la clasificación.
"""))
    celdas.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

print("✅ Librerías importadas correctamente.")

nombre_archivo = 'merge_RENAMU_GASTO_V.dta'
try:
    df = pd.read_stata(nombre_archivo)
    print(f"✅ Base de datos '{nombre_archivo}' cargada: {df.shape[0]} filas y {df.shape[1]} columnas.")
except FileNotFoundError:
    print(f"❌ Error: No se pudo encontrar el archivo '{nombre_archivo}'.")

mediana_contagiados = df['contagiados'].median()
df['nivel_contagios'] = df['contagiados'].apply(lambda x: 'ALTO' if x > mediana_contagiados else 'BAJO')
print("✅ Variable objetivo 'nivel_contagios' creada.")
"""))

    # ==========================================================================
    # --- PASO 2: PREPROCESAMIENTO Y LIMPIEZA DE DATOS ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
## Paso 2: Preprocesamiento y Limpieza de Datos

### 🎯 Objetivo de la Actividad
Preparar todas las variables del dataset para que sean aptas para el modelo, manejando valores faltantes, corrigiendo datos anómalos y asegurando un formato numérico.
"""))
    celdas.append(nbf.v4.new_code_cell("""
df_clean = df.copy()

# 1. Eliminar columnas no informativas o de fuga de datos
cols_to_drop = [
    'UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'VFI_P66', 
    'VFI_P67', 'VFI_P68', '_merge', 'P67_11_O', 'P68_8_O', 'contagiados'
]
df_clean = df_clean.drop(columns=cols_to_drop)

# 2. Corregir y transformar variables
df_clean.loc[df_clean['MONTO_GIRADO'] < 0, 'MONTO_GIRADO'] = 0
df_clean[['mes', 'year']] = df_clean[['mes', 'year']].apply(pd.to_numeric, errors='coerce')

# 3. Imputar valores faltantes
p_cols = [col for col in df_clean.columns if col.startswith('P')]
df_clean[p_cols] = df_clean[p_cols].fillna(0)

# 4. Estandarizar codificación binaria
p66_recode_2_to_0 = ['P66_1', 'P66_2', 'P66_3', 'P66_4', 'P66_5', 'P66_6', 'P66_7', 'P66_8', 'P66_9', 'P66_10']
for col in p66_recode_2_to_0:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(2, 0)

p_recode_gt0_to_1 = [col for col in df_clean.columns if col.startswith(('P67_', 'P68_'))]
for col in p_recode_gt0_to_1:
    df_clean[col] = df_clean[col].apply(lambda x: 1 if x > 0 else 0)

# 5. Definir X e y
y = df_clean['nivel_contagios'].apply(lambda nivel: 1 if nivel == 'ALTO' else 0)
X = df_clean.drop(columns=['nivel_contagios'])

print("✅ Preprocesamiento completado.")
print(f"Forma de X (predictoras): {X.shape}")
print(f"Forma de y (objetivo): {y.shape}")
"""))

    # ==========================================================================
    # --- PASO 3: DIVISIÓN DE DATOS (ENTRENAMIENTO Y PRUEBA) ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
## Paso 3: División de los Datos (Entrenamiento y Prueba)

### 🎯 Objetivo de la Actividad
Dividir nuestro conjunto de datos en dos subconjuntos: uno para entrenar el modelo y otro para evaluarlo de manera imparcial.

### ¿Por qué es crucial esta división?
- **Conjunto de Entrenamiento (Training Set)**: El modelo "aprende" los patrones y relaciones de estos datos. Típicamente, constituye el 70-80% del total.
- **Conjunto de Prueba (Test Set)**: Estos datos son "nuevos" para el modelo. Se usan para evaluar qué tan bien generaliza sus aprendizajes a datos que nunca ha visto. Esto nos da una medida realista de su rendimiento.

### 🧠 Conceptos Clave:
- **`train_test_split`**: La función de `scikit-learn` que realiza esta división.
- **`test_size`**: Define el porcentaje de datos que se destinará al conjunto de prueba (e.g., `0.2` para un 20%).
- **`random_state`**: Fija una "semilla" para la aleatoriedad, asegurando que la división sea siempre la misma cada vez que se ejecuta el código. Esto es vital para la **reproducibilidad**.
- **`stratify`**: Asegura que la proporción de clases (ej. "ALTO" vs "BAJO") sea la misma tanto en el conjunto de entrenamiento como en el de prueba. Es fundamental en problemas de clasificación para evitar sesgos.
"""))
    celdas.append(nbf.v4.new_code_cell("""
# Dividir los datos: 80% para entrenamiento, 20% para prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("✅ Datos divididos exitosamente.")
print(f"Tamaño del conjunto de entrenamiento (X_train): {X_train.shape}")
print(f"Tamaño del conjunto de prueba (X_test):      {X_test.shape}")
print(f"Proporción de clase 'ALTO' en y_train: {y_train.mean():.2f}")
print(f"Proporción de clase 'ALTO' en y_test:  {y_test.mean():.2f}")
"""))

    # ==========================================================================
    # --- PASO 4: ENTRENAMIENTO DEL MODELO RANDOM FOREST ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
## Paso 4: Entrenamiento del Modelo Random Forest

### 🎯 Objetivo de la Actividad
Instanciar y entrenar nuestro modelo de clasificación usando el conjunto de entrenamiento.

### ¿Qué es un Random Forest?
Un **Random Forest** (Bosque Aleatorio) es un modelo de *aprendizaje de conjunto* (ensemble learning). Funciona construyendo una multitud de **árboles de decisión** durante el entrenamiento y emitiendo la clase que es la moda de las clases (clasificación) o la predicción media (regresión) de los árboles individuales.

### 🧠 Conceptos Clave:
- **`RandomForestClassifier`**: La clase de `scikit-learn` que implementa el algoritmo.
- **`n_estimators`**: El número de árboles que se construirán en el bosque. Un número mayor generalmente mejora el rendimiento, pero también aumenta el costo computacional. `100` es un buen punto de partida.
- **`random_state`**: Al igual que antes, garantiza la reproducibilidad del modelo.
- **`fit(X_train, y_train)`**: El método que "entrena" el modelo, encontrando los patrones en los datos de entrenamiento.
"""))
    celdas.append(nbf.v4.new_code_cell("""
# 1. Crear una instancia del clasificador Random Forest.
#    n_jobs=-1 utiliza todos los núcleos de CPU disponibles para acelerar el entrenamiento.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Entrenar el modelo con los datos de entrenamiento.
print("🚀 Entrenando el modelo Random Forest...")
rf_model.fit(X_train, y_train)
print("✅ Modelo entrenado exitosamente.")
"""))

    # ==========================================================================
    # --- PASO 5: EVALUACIÓN DEL MODELO ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
## Paso 5: Evaluación del Modelo

### 🎯 Objetivo de la Actividad
Evaluar el rendimiento de nuestro modelo entrenado usando el conjunto de prueba, que contiene datos que el modelo no ha visto antes.

### Métricas Clave de Evaluación:
1.  **Matriz de Confusión**: Una tabla que muestra el rendimiento del modelo. Nos dice cuántos casos fueron clasificados correctamente y cuántos incorrectamente.
    - **Verdaderos Positivos (TP)**: Predijo "ALTO" y era "ALTO".
    - **Verdaderos Negativos (TN)**: Predijo "BAJO" y era "BAJO".
    - **Falsos Positivos (FP)**: Predijo "ALTO" pero era "BAJO" (Error Tipo I).
    - **Falsos Negativos (FN)**: Predijo "BAJO" pero era "ALTO" (Error Tipo II).

2.  **Reporte de Clasificación**:
    - **Accuracy (Exactitud)**: Porcentaje total de predicciones correctas. `(TP + TN) / Total`.
    - **Precision (Precisión)**: De todos los que predijo como "ALTO", ¿cuántos acertó? `TP / (TP + FP)`.
    - **Recall (Sensibilidad)**: De todos los que realmente eran "ALTO", ¿a cuántos identificó? `TP / (TP + FN)`.
    - **F1-Score**: La media armónica de Precisión y Recall. Es una métrica balanceada muy útil.

3.  **Curva ROC y AUC**:
    - **Curva ROC**: Visualiza la capacidad de un clasificador para distinguir entre clases. Un buen modelo tiene una curva que se acerca a la esquina superior izquierda.
    - **AUC (Area Under the Curve)**: El área bajo la curva ROC. Un valor de 1.0 representa un modelo perfecto, mientras que 0.5 representa un modelo que no es mejor que el azar.
"""))
    celdas.append(nbf.v4.new_code_cell("""
# 1. Hacer predicciones sobre el conjunto de prueba.
y_pred = rf_model.predict(X_test)

# 2. Calcular y mostrar la Matriz de Confusión.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred. BAJO', 'Pred. ALTO'], 
            yticklabels=['Real BAJO', 'Real ALTO'])
plt.title('Matriz de Confusión', fontsize=16)
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.show()

# 3. Imprimir el Reporte de Clasificación.
print("="*60)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['BAJO (0)', 'ALTO (1)']))
print("="*60)

# 4. Calcular y mostrar la Curva ROC y el AUC.
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
"""))

    # ==========================================================================
    # --- PASO 6: INTERPRETACIÓN DEL MODELO ---
    # ==========================================================================
    celdas.append(nbf.v4.new_markdown_cell("""
## Paso 6: Interpretación del Modelo

### 🎯 Objetivo de la Actividad
Entender qué variables fueron las más importantes para las predicciones del modelo. Un modelo no es solo una "caja negra"; Random Forest nos permite inspeccionar su lógica interna.

### Importancia de las Características (Feature Importance)
El algoritmo de Random Forest puede calcular una puntuación para cada variable predictora, indicando su contribución relativa a la reducción de la impureza (o mejora de la precisión) en los árboles del bosque.

Una puntuación más alta significa que la variable fue más decisiva para separar las clases "ALTO" y "BAJO".

**¿Para qué sirve esto?**
- **Entender el fenómeno**: Nos ayuda a comprender qué factores están más asociados con un alto nivel de contagios.
- **Selección de variables**: Podríamos decidir construir un modelo más simple usando solo las variables más importantes.
- **Comunicación**: Es una forma efectiva de explicar los resultados del modelo a partes interesadas no técnicas.
"""))
    celdas.append(nbf.v4.new_code_cell("""
# 1. Obtener la importancia de cada característica desde el modelo entrenado.
importances = rf_model.feature_importances_
feature_names = X.columns

# 2. Crear un DataFrame para facilitar la visualización.
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 3. Ordenar el DataFrame por importancia de forma descendente.
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 4. Visualizar las 20 características más importantes.
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
plt.title('Top 20 Variables más Importantes', fontsize=16)
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.grid(True, axis='x')
plt.show()

# Imprimir el top 10 para referencia
print("="*60)
print("Top 10 Variables más Importantes:")
print(feature_importance_df.head(10).to_string(index=False))
print("="*60)
"""))

    # Asignar todas las celdas al notebook
    nb.cells = celdas
    
    return nb

if __name__ == "__main__":
    # Generar el objeto notebook con todos los pasos
    notebook = generar_notebook_final()
    
    # Definir el nombre del archivo de salida
    notebook_filename = "Modelo_Random_Forest_Completo.ipynb"

    # Escribir el notebook a un archivo, asegurando la codificación UTF-8
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    
    print(f"✅ Notebook completo generado exitosamente!")
    print(f"📁 Archivo guardado como: {notebook_filename}")
    print("🚀 Puede abrir el notebook en Jupyter, VS Code, o Google Colab.")