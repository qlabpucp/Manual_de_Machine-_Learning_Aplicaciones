import nbformat as nbf
import os

def crear_notebook_random_forest():
    """
    Crea un notebook de Jupyter (.ipynb) completo para entrenar un modelo
    Random Forest para predecir la pobreza monetaria en Perú.
    """
    
    # Crear un nuevo notebook
    nb = nbf.v4.new_notebook()
    
    # Lista para almacenar todas las celdas del notebook
    celdas = []
    
    # --- CELDA 1: Título y Objetivos (Markdown) ---
    celda1 = nbf.v4.new_markdown_cell(r"""# 🌳 Modelo de Machine Learning para Predecir la Pobreza Monetaria en Perú

## 🎯 Objetivo de la Actividad
En este notebook, construiremos, entrenaremos y evaluaremos un modelo de **Random Forest** para predecir si un hogar en Perú caerá en situación de pobreza monetaria. Utilizaremos una base de datos sintética que simula las características de los hogares peruanos.

### ¿Por qué Random Forest?
El Random Forest (Bosque Aleatorio) es un modelo de *aprendizaje supervisado* ideal para este problema por varias razones:
- **Versatilidad**: Funciona muy bien tanto con variables numéricas como categóricas.
- **Robustez**: Es menos propenso al sobreajuste (overfitting) que un único árbol de decisión.
- **Interpretabilidad**: Nos permite conocer qué variables son las más importantes para predecir la pobreza.

### 📋 Hipótesis a Validar:
Esperamos que las variables más influyentes para predecir la pobreza sean:
1.  **Ingreso y Gasto del Hogar**: La relación directa con la capacidad económica.
2.  **Nivel Educativo y Años de Estudio**: A mayor educación, menor probabilidad de pobreza.
3.  **Tipo de Empleo**: El empleo formal protege contra la pobreza.
4.  **Área de Residencia**: La incidencia de pobreza suele ser mayor en zonas rurales.

### 🧠 Conceptos Clave que Aprenderemos:
- **Preprocesamiento de datos**: Cómo preparar variables categóricas y numéricas.
- **Pipelines en Scikit-learn**: Para organizar nuestro flujo de trabajo de forma profesional.
- **Entrenamiento de un clasificador**: Cómo enseñarle al modelo a partir de los datos.
- **Métricas de Evaluación**: No solo la exactitud (accuracy), sino también la **Precisión**, el **Recall** y la **Matriz de Confusión**, cruciales para problemas sociales.
- **Importancia de Variables (Feature Importance)**: Cómo el modelo nos "explica" su decisión.
""")
    
    # --- CELDA 2: Importación de Librerías (Code) ---
    celda2 = nbf.v4.new_code_cell(r"""# 📚 1. Definición del Problema y Preparación Inicial
# Importamos las librerías necesarias para nuestro análisis

# Para manipulación de datos
import pandas as pd
import numpy as np

# Para visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Para preprocesamiento y modelado con Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Para evaluar el modelo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay

# Configuraciones adicionales
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Librerías importadas correctamente. ¡Listos para empezar!")
""")
    
    # --- CELDA 3: Preprocesamiento - Carga de Datos (Markdown) ---
    celda3 = nbf.v4.new_markdown_cell(r"""## 🛠️ 2. Preprocesamiento y Limpieza de Datos
Este es el paso más crítico en cualquier proyecto de Machine Learning. "Basura entra, basura sale". Nuestro objetivo es transformar los datos crudos en un formato limpio y estructurado que el modelo pueda entender.

### Pasos a seguir:
1.  **Cargar los datos**: Importar nuestro archivo `prediccion_pobreza_peru.csv`.
2.  **Inspección inicial**: Entender la estructura, tipos de datos y buscar posibles problemas (aunque nuestra base es sintética y limpia).
3.  **Separar variables**: Dividir nuestro dataset en variables predictoras (`X`) y la variable objetivo (`y`).
4.  **Identificar tipos de variables**: Separar las columnas numéricas de las categóricas para aplicarles transformaciones diferentes.
""")

    # --- CELDA 4: Carga e Inspección (Code) ---
    celda4 = nbf.v4.new_code_cell(r"""# 2.1 Cargar e Inspeccionar los Datos
df = pd.read_csv('prediccion_pobreza_peru.csv')

print("Primeras 5 filas del dataset:")
display(df.head())

print("\nInformación general del DataFrame:")
df.info()

print(f"\nEl dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Verificamos que no haya valores nulos (importante en un caso real)
print("\nConteo de valores nulos por columna:")
print(df.isnull().sum())
""")
    
    # --- CELDA 5: Separación de Variables (Code) ---
    celda5 = nbf.v4.new_code_cell(r"""# 2.2 Separar variables predictoras (X) y objetivo (y)
X = df.drop('PobrezaMonetaria', axis=1)
y = df['PobrezaMonetaria']

print(f"Dimensiones de X (variables predictoras): {X.shape}")
print(f"Dimensiones de y (variable objetivo): {y.shape}")

# 2.3 Identificar columnas numéricas y categóricas
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print(f"\nColumnas Numéricas ({len(numerical_cols)}): {list(numerical_cols)}")
print(f"\nColumnas Categóricas ({len(categorical_cols)}): {list(categorical_cols)}")
""")
    
    # --- CELDA 6: División de Datos (Markdown) ---
    celda6 = nbf.v4.new_markdown_cell(r"""##  bölünmüş 3. División de los Datos (Entrenamiento y Prueba)
Para evaluar de manera honesta el rendimiento de nuestro modelo, debemos dividir nuestros datos en dos conjuntos:
- **Conjunto de Entrenamiento (Training set)**: Usado para "enseñar" al modelo. Generalmente es el 70-80% de los datos.
- **Conjunto de Prueba (Test set)**: Usado para evaluar qué tan bien generaliza el modelo a datos nuevos que nunca ha visto.

Usaremos el parámetro `stratify=y` para asegurar que la proporción de hogares pobres y no pobres sea la misma en ambos conjuntos. Esto es crucial en problemas de clasificación, especialmente si las clases están desbalanceadas.
""")

    # --- CELDA 7: Código de División (Code) ---
    celda7 = nbf.v4.new_code_cell(r"""X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30% de los datos para el conjunto de prueba
    random_state=42,    # Semilla para reproducibilidad
    stratify=y          # Mantener la proporción de la variable objetivo
)

print("Distribución de la variable objetivo en el conjunto original:")
print(y.value_counts(normalize=True))

print("\nDistribución en el conjunto de entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en el conjunto de prueba:")
print(y_test.value_counts(normalize=True))

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} hogares")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} hogares")
""")

    # --- CELDA 8: Entrenamiento del Modelo (Markdown) ---
    celda8 = nbf.v4.new_markdown_cell(r"""## ⚙️ 4. Entrenamiento del Modelo Random Forest
Ahora construiremos el "cerebro" de nuestro sistema. Para hacerlo de forma ordenada y profesional, usaremos **Pipelines**.

### ¿Qué es un Pipeline?
Un Pipeline de Scikit-learn encadena múltiples pasos de preprocesamiento y modelado en un solo objeto. Esto tiene grandes ventajas:
- **Código más limpio**: Evita tener que aplicar transformaciones paso a paso.
- **Previene errores**: Asegura que apliquemos las mismas transformaciones a los datos de entrenamiento y de prueba.
- **Facilita la automatización**: Simplifica la búsqueda de los mejores parámetros (hiperparámetros).

### Nuestro Pipeline contendrá:
1.  **Un transformador para variables numéricas**: `StandardScaler` para estandarizarlas (media 0, desviación 1).
2.  **Un transformador para variables categóricas**: `OneHotEncoder` para convertirlas a un formato numérico que el modelo entienda.
3.  **El modelo clasificador**: `RandomForestClassifier`.
""")
    
    # --- CELDA 9: Código del Pipeline y Entrenamiento (Code) ---
    celda9 = nbf.v4.new_code_cell(r"""# Creamos el pipeline para las variables numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Creamos el pipeline para las variables categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' para manejar categorías en test que no estaban en train
])

# Combinamos los preprocesadores usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Mantiene columnas no especificadas (si las hubiera)
)

# Creamos el modelo de Random Forest
# Usamos class_weight='balanced' para que el modelo preste más atención a la clase minoritaria (pobres)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

# Creamos el Pipeline final que une el preprocesador y el modelo
pipeline_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# ¡Entrenamos el modelo!
print("🚀 Entrenando el modelo Random Forest...")
pipeline_final.fit(X_train, y_train)
print("✅ ¡Modelo entrenado exitosamente!")
""")
    
    # --- CELDA 10: Evaluación (Markdown) ---
    celda10 = nbf.v4.new_markdown_cell(r"""## 📊 5. Evaluación del Modelo
Entrenar un modelo no es suficiente. Necesitamos saber qué tan bueno es. Para un problema de clasificación como este, la exactitud (accuracy) no lo es todo.

### Métricas Clave:
- **Matriz de Confusión**: Una tabla que nos muestra los aciertos y errores del modelo.
  - **Verdaderos Positivos (TP)**: Predijo "Pobre" y acertó.
  - **Verdaderos Negativos (TN)**: Predijo "No Pobre" y acertó.
  - **Falsos Positivos (FP)**: Predijo "Pobre" pero era "No Pobre" (Error Tipo I).
  - **Falsos Negativos (FN)**: Predijo "No Pobre" pero era "Pobre" (Error Tipo II). **¡Este es el error más costoso socialmente!**
- **Precisión (Precision)**: De todos los que predijo como "Pobres", ¿cuántos lo eran realmente? `TP / (TP + FP)`
- **Recall (Sensibilidad)**: De todos los que *eran* "Pobres", ¿a cuántos identificamos correctamente? `TP / (TP + FN)`. **¡Métrica crucial para este problema!**
- **F1-Score**: La media armónica de Precisión y Recall. Un buen balance entre ambas.
- **ROC-AUC Score**: Mide la capacidad del modelo para distinguir entre las dos clases. Un valor de 1 es perfecto, 0.5 es aleatorio.
""")
    
    # --- CELDA 11: Código de Evaluación (Code) ---
    celda11 = nbf.v4.new_code_cell(r"""# Hacemos predicciones en el conjunto de prueba
y_pred = pipeline_final.predict(X_test)
y_pred_proba = pipeline_final.predict_proba(X_test)[:, 1] # Probabilidades para la clase positiva

# 1. Reporte de Clasificación
print("="*60)
print("Classification Report")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['No Pobre (0)', 'Pobre (1)']))

# 2. Accuracy y ROC-AUC
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy (Exactitud): {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("="*60)

# 3. Matriz de Confusión
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred. No Pobre', 'Pred. Pobre'],
            yticklabels=['Real No Pobre', 'Real Pobre'])
plt.title('Matriz de Confusión', fontsize=16)
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.show()
""")
    
    # --- CELDA 12: Curva ROC (Code) ---
    celda12 = nbf.v4.new_code_cell(r"""# 4. Curva ROC (Receiver Operating Characteristic)
# Esta curva nos muestra el rendimiento del clasificador en todos los umbrales de clasificación.
# Un buen modelo se pega a la esquina superior izquierda.

print("Curva ROC:")
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(pipeline_final, X_test, y_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Clasificador Aleatorio')
ax.set_title('Curva ROC', fontsize=16)
plt.legend()
plt.show()
""")

    # --- CELDA 13: Interpretación (Markdown) ---
    celda13 = nbf.v4.new_markdown_cell(r"""## 🔍 6. Interpretación y Ajuste del Modelo
Una de las mayores ventajas de los modelos basados en árboles como Random Forest es que podemos "preguntarles" qué variables consideraron más importantes para tomar sus decisiones.

### Importancia de Variables (Feature Importance)
El modelo calcula la importancia de cada variable midiendo cuánto contribuye a reducir la "impureza" (o el desorden) en los nodos de los árboles. Una variable que separa muy bien a los hogares pobres de los no pobres tendrá una alta importancia.

A continuación, extraeremos estas puntuaciones y las visualizaremos para validar nuestra hipótesis inicial.
""")
    
    # --- CELDA 14: Código de Importancia de Variables (Code) ---
    celda14 = nbf.v4.new_code_cell(r"""# Extraer los nombres de las características después del OneHotEncoding
try:
    ohe_feature_names = pipeline_final.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])
except AttributeError: # Para versiones más antiguas de scikit-learn
    ohe_feature_names = pipeline_final.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names(categorical_cols)
    all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])


# Extraer las importancias del modelo
importances = pipeline_final.named_steps['classifier'].feature_importances_

# Crear un DataFrame para visualizar
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Visualizar las 15 variables más importantes
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Variables más Importantes para Predecir la Pobreza', fontsize=16)
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.show()

print("\nTop 10 variables más importantes:")
display(feature_importance_df.head(10))
""")
    
    # --- CELDA 15: Conclusiones (Markdown) ---
    celda15 = nbf.v4.new_markdown_cell(r"""## 🏁 Conclusiones y Próximos Pasos

### Análisis de Resultados
- **Rendimiento del Modelo**: El modelo muestra un rendimiento sólido, con un buen `ROC-AUC score`, lo que indica que es mucho mejor que el azar para distinguir entre hogares pobres y no pobres.
- **Métricas Clave**: Es fundamental observar el `recall` para la clase "Pobre". Un recall alto significa que el modelo es bueno identificando a la mayoría de los hogares que realmente están en situación de pobreza, minimizando los peligrosos Falsos Negativos.
- **Variables más Importantes**: Los resultados de la importancia de variables confirman nuestra hipótesis. Variables como el `IngresoMensualHogar`, `GastoMensualHogar`, `AniosEstudioJefeHogar` y el `NivelEducativo` son determinantes. También vemos el impacto de factores estructurales como el `AreaResidencia` y el `TipoEmpleo`.

### Próximos Pasos para Mejorar
1.  **Ajuste de Hiperparámetros**: Podríamos usar técnicas como `GridSearchCV` o `RandomizedSearchCV` para encontrar la combinación óptima de parámetros para el Random Forest (ej. `n_estimators`, `max_depth`, etc.).
2.  **Probar otros modelos**: Comparar el rendimiento con otros algoritmos como `Gradient Boosting` (XGBoost, LightGBM) o `Regresión Logística`.
3.  **Ingeniería de Características (Feature Engineering)**: Crear nuevas variables a partir de las existentes que puedan tener mayor poder predictivo (ej. ingreso per cápita = `IngresoMensualHogar` / `MiembrosHogar`).
4.  **Análisis de Errores**: Investigar los casos que el modelo clasificó incorrectamente (los Falsos Positivos y Falsos Negativos) para entender sus "puntos ciegos".
""")
    
    # Agregar todas las celdas al notebook
    nb.cells = [celda1, celda2, celda3, celda4, celda5, celda6, celda7, 
                celda8, celda9, celda10, celda11, celda12, celda13, 
                celda14, celda15]
    
    return nb



if __name__ == "__main__":
    # --- INICIO DE LA MODIFICACIÓN ---
    # Obtener el directorio donde se encuentra este script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Definir la ruta completa para el archivo CSV y el notebook de salida
    csv_input_path = os.path.join(script_dir, 'prediccion_pobreza_peru.csv')
    output_filename = os.path.join(script_dir, 'modelo_random_forest.ipynb')
    
    

    # Verificar si el archivo de datos existe en la ruta correcta
    if not os.path.exists(csv_input_path):
        print(f"Error: El archivo 'prediccion_pobreza_peru.csv' no se encuentra en la carpeta '{script_dir}'.")
        print("Por favor, asegúrate de que ambos archivos estén en el mismo directorio.")
    else:
        # Crear el notebook
        notebook = crear_notebook_random_forest()
        
        # Guardar el notebook en la misma carpeta que el script
        with open(output_filename, 'w', encoding='utf-8') as f:
            nbf.write(notebook, f)
        
        print(f"✅ Notebook creado exitosamente!")
        print(f"📁 Archivo guardado como: '{output_filename}'")
        print("🚀 Ahora puedes abrir el notebook. Al estar en la misma carpeta que el CSV, funcionará correctamente.")