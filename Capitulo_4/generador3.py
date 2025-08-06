import nbformat as nbf

def generar_notebook_modelo_formato_corregido():
    """
    Genera un notebook de Jupyter con el formato de Markdown corregido
    para un manual introductorio, demostrando la Regresión Lasso.
    """
    
    print("Iniciando la creación del notebook de modelado con formato corregido...")
    
    nb = nbf.v4.new_notebook()

    # --- CELDA 1: Título e Introducción (Markdown) ---
    cell1_md = """
# # Aplicación Práctica: Predicción de Ingresos con Regresión Lasso
# 
## 🎯 Objetivo de la Actividad
En esta actividad práctica, aprenderemos a usar la **Regresión Lasso** para predecir el ingreso monetario de una persona. Más allá de la predicción, nuestro objetivo principal es descubrir qué características socioeconómicas son las más influyentes para determinar dicho ingreso.

### ¿Qué es la Regresión Lasso?
Es una técnica de Machine Learning que realiza dos tareas simultáneamente:
1.  **Predicción**: Crea un modelo para estimar un valor numérico (como el ingreso).
2.  **Selección de Variables**: Identifica y descarta automáticamente las variables menos importantes.

### 📋 Hipótesis que vamos a probar:
1.  Podemos construir un modelo que explique una parte significativa de la variación en los ingresos.
2.  **Lasso** nos ayudará a identificar un subconjunto de variables clave de entre todas las disponibles en la encuesta.

### 🧠 Conceptos Clave que Aprenderemos:
- **Regresión Lineal**: La base sobre la que se construye Lasso.
- **Penalización L1**: El "ingrediente secreto" de Lasso que permite la selección de variables.
- **Validación Cruzada**: Cómo elegir el mejor parámetro para nuestro modelo de forma automática.
- **Interpretación de Coeficientes**: Cómo entender los resultados del modelo y qué nos dicen sobre el mundo real.
"""

    # --- CELDA 2: Configuración del Entorno (Código) ---
    cell2_code = """
# 📚 Importar librerías necesarias
# NumPy: Para operaciones matemáticas y arrays
import numpy as np

# Pandas: Para manipulación y análisis de datos
import pandas as pd

# Matplotlib y Seaborn: Para visualizaciones de alta calidad
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Nuestra librería principal de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

#  Configurar estilo de gráficos para que se vean más profesionales
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

print("✅ Librerías importadas correctamente")
print("📊 Configuración de gráficos lista")
print("🚀 ¡Listos para comenzar la actividad!")
"""

    # --- CELDA 3: Carga de Datos (Markdown) ---
    cell3_md = """
## Paso 1: Carga y Exploración de los Datos

### ¿Qué datos estamos usando?
Vamos a cargar un archivo llamado `data_processed_income_positive.csv`. Este no es el archivo original de la encuesta, sino una versión que ya ha sido preparada para el modelado.
 
### 🧠 Concepto Importante: Preprocesamiento
El preprocesamiento es el conjunto de pasos que se realizan para limpiar y transformar los datos antes de dárselos a un modelo. En nuestro caso, el preprocesamiento incluyó:

1.  **Filtrado**: Se eliminaron las personas con ingreso cero para simplificar el problema.
2.  **Transformación Logarítmica**: La variable de ingreso fue transformada para que su distribución sea más simétrica, lo que ayuda a los modelos lineales.
3.  **Estandarización y Codificación**: Las variables numéricas se pusieron en la misma escala y las categóricas se convirtieron a un formato numérico.

> 💡 **Nota:** La calidad del preprocesamiento es a menudo más importante que la elección del modelo en sí.
"""
    
    # --- CELDA 4: Carga de Datos (Código) ---
    cell4_code = """
# 📂 Cargar los datos ya procesados desde el archivo CSV
try:
    data = pd.read_csv("data_processed_income_positive.csv")
    print(f"✅ Datos preprocesados cargados exitosamente.")
    print(f"📊 Dimensiones del dataset: {data.shape[0]} filas y {data.shape[1]} columnas.")
except FileNotFoundError:
    print("❌ Error: El archivo 'data_processed_income_positive.csv' no fue encontrado.")
    print("   Por favor, asegúrate de haber ejecutado primero el script de preprocesamiento.")

# 📋 Mostrar las primeras 5 filas para verificar la carga
print("\\n📋 Primeras 5 filas del dataset procesado:")
display(data.head())
"""

    # --- CELDA 5: Preparación de Datos (Markdown) ---
    cell5_md = """
## Paso 2: Preparación para el Modelado

Antes de entrenar, debemos realizar dos pasos cruciales:

1.  **Separar variables**: Distinguir entre las **variables predictoras (X)** y la **variable objetivo (y)**.
2.  **Dividir en entrenamiento y prueba**: Para evaluar de forma honesta qué tan bien generaliza nuestro modelo.

### 🧠 Conceptos Importantes:

**Variables Predictoras (X) vs. Variable Objetivo (y)**
- **X (features)**: Todas las columnas excepto `log_ingmo2hd`.
- **y (target)**: La columna `log_ingmo2hd`.

**Train-Test Split**
- **Datos de Entrenamiento (70%)**: Para "enseñarle" al modelo los patrones en los datos.
- **Datos de Prueba (30%)**: Para evaluar el rendimiento del modelo en datos "nuevos" que no ha visto durante el entrenamiento.
"""

    # --- CELDA 6: Separación y División (Código) ---
    cell6_code = """
# 📊 Separar variables explicativas (X) y variable objetivo (y)
print("🔍 Separando variables predictoras (X) y objetivo (y)...")
X = data.drop('log_ingmo2hd', axis=1)
y = data['log_ingmo2hd']

print(f"📈 Variables predictoras (X): {X.shape[1]} columnas")
print(f"🎯 Variable objetivo (y): 1 columna ('log_ingmo2hd')")

# 🔄 Dividir en conjuntos de entrenamiento y prueba
print("\\n🔄 Dividiendo datos en entrenamiento (70%) y prueba (30%)...")
# 'random_state=42' es una semilla que asegura que la división sea siempre la misma, haciendo nuestro experimento reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"📚 Datos de entrenamiento: {X_train.shape[0]} observaciones")
print(f"🧪 Datos de prueba: {X_test.shape[0]} observaciones")

print("\\n✅ ¡Datos preparados y listos para entrenar el modelo!")
"""

    # --- CELDA 7: Entrenamiento del Modelo (Markdown) ---
    cell7_md = """
## Paso 3: Entrenamiento del Modelo Lasso
 
### ¿Qué es `LassoCV`?
Usaremos `LassoCV`. Es una versión "inteligente" de Lasso que incluye **Validación Cruzada (Cross-Validation)** para encontrar el mejor parámetro de regularización de forma automática.

### 🧠 Concepto clave: El Parámetro `alpha`
Lasso tiene un parámetro llamado **alpha (α)** que controla la fuerza de la regularización:
- **α pequeño**: Poca penalización. El modelo se parece a una regresión normal.
- **α grande**: Mucha penalización. El modelo elimina más variables.

`LassoCV` prueba una gama de `alphas` y elige el que produce el mejor rendimiento predictivo.
"""

    # --- CELDA 8: Entrenamiento del Modelo (Código) ---
    cell8_code = """
# 🚀 Entrenar el modelo Lasso con Validación Cruzada
print("🔍 Buscando el mejor parámetro alpha y entrenando el modelo Lasso...")

# 1. Definimos el modelo LassoCV
#    Le decimos que pruebe 200 valores de alpha y use 5 "folds" para la validación cruzada.
lasso_cv_model = LassoCV(alphas=np.logspace(-5, 1, 200), cv=5, random_state=42, max_iter=2000)

# 2. Entrenamos el modelo con los datos de entrenamiento
lasso_cv_model.fit(X_train, y_train)

# 🏆 Mostrar el mejor alpha encontrado
print(f"\\n🏆 Mejor alpha encontrado: {lasso_cv_model.alpha_:.5f}")
print("✅ ¡Modelo Lasso entrenado exitosamente!")
"""

    # --- CELDA 9: Evaluación (Markdown) ---
    cell9_md = """
## Paso 4: Evaluación del Desempeño del Modelo
 
Ahora que el modelo está entrenado, debemos evaluar qué tan bien funciona en el conjunto de prueba.

### 🧠 Métricas de Evaluación:
- **RMSE (Raíz del Error Cuadrático Medio)**: Nos dice, en promedio, cuál es el error de predicción en la unidad original (Soles). **Menor es mejor**.
- **R² (Coeficiente de Determinación)**: Indica qué porcentaje de la variabilidad del ingreso es explicado por nuestro modelo. **Más cercano a 1 (o 100%) es mejor**.

### 💡 Paso Crucial: Revertir la Transformación Logarítmica
Nuestro modelo predice el *logaritmo* del ingreso. Para que el error (RMSE) sea interpretable, debemos convertir las predicciones de vuelta a Soles usando la función exponencial (`np.expm1`).
"""

    # --- CELDA 10: Evaluación (Código) ---
    cell10_code = """
# 📈 Hacer predicciones en datos de prueba
y_pred_log = lasso_cv_model.predict(X_test)

# 🔄 Revertir la transformación logarítmica para obtener predicciones en Soles
y_pred_original = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# 📊 Calcular métricas de rendimiento
rmse_lasso = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2_lasso = r2_score(y_test_original, y_pred_original)

print("📊 RESULTADOS DEL MODELO LASSO EN DATOS DE PRUEBA:")
print(f"💰 RMSE (Error de predicción): ${rmse_lasso:,.2f} Soles")
print(f"📈 R² (Coeficiente de determinación): {r2_lasso:.4f} (es decir, el modelo explica el {r2_lasso:.1%} de la varianza del ingreso)")
"""

    # --- CELDA 11: Interpretación (Markdown) ---
    cell11_md = """
## Paso 5: Interpretación de Resultados - El Poder de Lasso
 
Esta es la parte más interesante. ¿Qué variables consideró el modelo como importantes?

### 🧠 Concepto Clave: Coeficientes del Modelo
El modelo asigna un **coeficiente** a cada variable. La magia de Lasso es que:
- Si el coeficiente es **cero**, Lasso ha **descartado** esa variable por no considerarla relevante.
- Si el coeficiente es **distinto de cero**, es una **variable seleccionada**.

**Interpretación de los coeficientes (en nuestro modelo log-level):**
- Un **coeficiente positivo** (ej: 0.10) significa que un aumento en esa variable se asocia con un aumento porcentual en el ingreso (aprox. 10%).
- Un **coeficiente negativo** (ej: -0.05) significa que un aumento en esa variable se asocia con una disminución porcentual en el ingreso (aprox. 5%).
"""
    
    # --- CELDA 12: Interpretación (Código) ---
    cell12_code = """
# 🔍 Analizar coeficientes y selección de variables
print("🔍 Analizando los coeficientes del modelo Lasso...")

# Crear un DataFrame con los predictores y sus coeficientes
coeficientes_lasso = pd.DataFrame({
    'predictor': X.columns,
    'coef': lasso_cv_model.coef_
})

# Filtrar los coeficientes que no son cero
coeficientes_seleccionados = coeficientes_lasso[coeficientes_lasso['coef'] != 0].copy()
lasso_zero = np.sum(lasso_cv_model.coef_ == 0)

# Ordenar por el valor absoluto para ver los más importantes
coeficientes_seleccionados['importancia'] = coeficientes_seleccionados['coef'].abs()
coeficientes_seleccionados = coeficientes_seleccionados.sort_values(by='importancia', ascending=False).drop('importancia', axis=1)

print(f"\\n🎯 De {len(coeficientes_lasso)} características iniciales, Lasso seleccionó {len(coeficientes_seleccionados)}.")
print(f"❌ Lasso eliminó {lasso_zero} variables al asignarles un coeficiente de cero.")

print("\\n📋 Variables más importantes seleccionadas por el modelo:")
display(coeficientes_seleccionados.head(15).round(4))
"""
    
    # --- CELDA 13: Visualización (Markdown) ---
    cell13_md = """
## Paso 6: Visualización de los Coeficientes
 
Un gráfico a menudo cuenta una historia más clara que una tabla. Vamos a visualizar los 20 coeficientes más importantes para entender de un vistazo qué factores tienen el mayor impacto en el ingreso.

### 📈 Lo que vamos a observar:
- **Barras a la derecha (positivas)**: Características que aumentan el ingreso.
- **Barras a la izquierda (negativas)**: Características que disminuyen el ingreso.
- **Longitud de la barra**: La magnitud del impacto.
"""

    # --- CELDA 14: Visualización (Código) ---
    cell14_code = """
# 🎨 Preparando la visualización de coeficientes...

# Tomamos los 20 coeficientes con mayor valor absoluto y los ordenamos por su valor para el gráfico
top_coef = coeficientes_seleccionados.head(20).sort_values('coef')

plt.figure(figsize=(14, 10))
sns.barplot(x='coef', y='predictor', data=top_coef, palette='coolwarm')

plt.title('Top 20 Coeficientes del Modelo Lasso', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Valor del Coeficiente (Impacto en el Log-Ingreso)', fontsize=14)
plt.ylabel('Predictor', fontsize=14)
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
"""
    # --- CELDA 15: Conclusión (Markdown) ---
    cell15_md = """
## 🎓 Resumen y Conclusiones de la Actividad
 
### 🧠 Lo que hemos logrado:
1.  **Construimos un Modelo Predictivo**: Creamos un modelo Lasso que puede estimar el ingreso de una persona basándose en sus características, explicando una porción significativa de la varianza (`R²`).
2.  **Realizamos Selección de Variables**: De un gran número de variables iniciales, Lasso nos ayudó a identificar un subconjunto más pequeño y manejable de predictores importantes. Esto es invaluable en ciencias sociales, donde a menudo tenemos "demasiados" datos.
3.  **Obtuvimos Insights Interpretables**: Al analizar los coeficientes, podemos empezar a formular hipótesis sobre qué factores (como la educación, el tipo de trabajo, o la posesión de ciertos bienes) están más fuertemente asociados con mayores o menores ingresos en nuestra población de estudio.

### 💡 Conceptos clave para recordar:
- **Lasso (L1 Regularization)**: Es una herramienta poderosa para construir modelos *parsimoniosos* (simples) y evitar el sobreajuste.
- **Preprocesamiento**: Es un paso no negociable. La calidad del modelo depende directamente de la calidad de los datos que le damos.
- **Interpretación**: Los modelos no son solo cajas negras. Entender los coeficientes nos permite conectar los resultados matemáticos con el conocimiento del dominio.

### 🎉 ¡Felicidades!
Has completado exitosamente esta actividad. Ahora tienes una comprensión sólida de cómo aplicar la Regresión Lasso a un problema del mundo real, desde la carga de datos hasta la interpretación de resultados.

¡Sigue practicando y explorando para afianzar estos conceptos! 🚀
"""

    notebook_cells = [
        nbf.v4.new_markdown_cell(cell1_md),
        nbf.v4.new_code_cell(cell2_code),
        nbf.v4.new_markdown_cell(cell3_md),
        nbf.v4.new_code_cell(cell4_code),
        nbf.v4.new_markdown_cell(cell5_md),
        nbf.v4.new_code_cell(cell6_code),
        nbf.v4.new_markdown_cell(cell7_md),
        nbf.v4.new_code_cell(cell8_code),
        nbf.v4.new_markdown_cell(cell9_md),
        nbf.v4.new_code_cell(cell10_code),
        nbf.v4.new_markdown_cell(cell11_md),
        nbf.v4.new_code_cell(cell12_code),
        nbf.v4.new_markdown_cell(cell13_md),
        nbf.v4.new_code_cell(cell14_code),
        nbf.v4.new_markdown_cell(cell15_md)
    ]

    nb['cells'] = notebook_cells
    notebook_filename = "Modelo_Lasso_Prediccion_Ingreso.ipynb"
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"\nNotebook '{notebook_filename}' creado exitosamente con el formato corregido.")

if __name__ == "__main__":
    generar_notebook_modelo_formato_corregido()