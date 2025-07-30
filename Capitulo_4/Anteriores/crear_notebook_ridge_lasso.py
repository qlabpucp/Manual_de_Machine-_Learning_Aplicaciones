
import nbformat as nbf

def crear_notebook_ridge_lasso():
    """Crear un notebook completo para comparar Ridge vs Lasso"""
    
    # Crear un nuevo notebook
    nb = nbf.v4.new_notebook()
    
    # Lista de todas las celdas
    celdas = []
    
    # === CELDA 1: Título principal ===
    celda1 = nbf.v4.new_markdown_cell("""# Actividad: Lasso vs. Ridge - Comparación de Modelos de Regularización

## 🎯 Objetivo de la Actividad
En esta actividad práctica, aprenderemos a comparar dos técnicas fundamentales de regularización en Machine Learning: **Ridge Regression** y **Lasso Regression**. 

### ¿Qué es la Regularización?
La regularización es una técnica que ayuda a prevenir el **overfitting** (sobreajuste) en nuestros modelos. Cuando un modelo se sobreajusta, memoriza los datos de entrenamiento pero no generaliza bien a nuevos datos.

### ¿Por qué comparar Ridge vs Lasso?
- **Ridge (L2)**: Reduce los coeficientes pero nunca los hace exactamente cero
- **Lasso (L1)**: Puede hacer que algunos coeficientes sean exactamente cero, eliminando variables

### 📋 Hipótesis que vamos a probar:
1. **Lasso** tendrá un error de predicción similar o mejor que Ridge
2. **Lasso** producirá un modelo más interpretable al reducir a cero los coeficientes de variables irrelevantes
3. **Ridge** mantendrá todos los coeficientes pero con valores pequeños

### 🧠 Conceptos Clave que Aprenderemos:
- **Penalización L1 vs L2**: Diferentes formas de regularizar
- **Selección de Variables**: Cómo Lasso puede eliminar automáticamente variables irrelevantes
- **Trade-off**: Interpretabilidad vs Rendimiento predictivo
- **Validación Cruzada**: Para encontrar el mejor parámetro de regularización""")
    
    # === CELDA 2: Importaciones ===
    celda2 = nbf.v4.new_code_cell("""# 📚 Importar librerías necesarias
# NumPy: Para operaciones matemáticas y arrays
import numpy as np

# Pandas: Para manipulación y análisis de datos
import pandas as pd

# Matplotlib y Seaborn: Para visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Librería principal de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración para evitar warnings innecesarios
import warnings
warnings.filterwarnings('ignore')

# 🎨 Configurar estilo de gráficos para que se vean más bonitos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Librerías importadas correctamente")
print("📊 Configuración de gráficos lista")
print("🚀 ¡Listos para comenzar la actividad!")""")
    
    # === CELDA 3: Sección de datos simulados ===
    celda3 = nbf.v4.new_markdown_cell("""## 📊 Paso 1: Generación de Datos Simulados

### ¿Por qué usar datos simulados?
Para entender mejor cómo funcionan Ridge y Lasso, vamos a crear datos donde **sabemos exactamente** qué variables son importantes y cuáles no. Esto nos permitirá evaluar si nuestros modelos pueden identificar correctamente las variables relevantes.

### 🎯 Estructura de nuestros datos:
- **10 variables relevantes**: Que realmente afectan el ingreso anual
- **20 variables irrelevantes**: Que no tienen relación con el ingreso
- **1000 observaciones**: Para tener suficientes datos
- **Ruido**: Para simular condiciones reales (nada es perfecto en la vida real)

### 🧠 Concepto importante: Coeficientes Verdaderos
En la vida real, nunca sabemos los "coeficientes verdaderos", pero aquí los definimos para poder evaluar qué tan bien funcionan nuestros modelos.

### 📈 Variables que simularemos:
1. **Educación** (coef = 5000): Años de estudio
2. **Experiencia laboral** (coef = 3000): Años trabajando
3. **Edad** (coef = 2000): Edad del trabajador
4. **Horas trabajadas** (coef = 1500): Horas semanales
5. **Sector económico** (coef = 1000): Tipo de industria
6. **Tamaño de empresa** (coef = 800): Número de empleados
7. **Nivel de responsabilidad** (coef = 600): Cargo en la empresa
8. **Ubicación geográfica** (coef = 400): Ciudad/región
9. **Certificaciones** (coef = 300): Certificaciones profesionales
10. **Idiomas** (coef = 200): Número de idiomas hablados""")
    
    # === CELDA 4: Código de generación de datos ===
    celda4 = nbf.v4.new_code_cell("""# 🔧 Configurar semilla para reproducibilidad
# Esto asegura que obtengamos los mismos resultados cada vez que ejecutemos el código
np.random.seed(42)

# 📊 Parámetros de la simulación
n_samples = 1000                    # Número de personas en nuestro dataset
n_relevant_features = 10            # Variables que realmente afectan el ingreso
n_irrelevant_features = 20          # Variables que NO afectan el ingreso
n_total_features = n_relevant_features + n_irrelevant_features

print(f"🎯 Creando dataset con {n_samples} personas y {n_total_features} variables")
print(f"📈 Variables relevantes: {n_relevant_features}")
print(f"❌ Variables irrelevantes: {n_irrelevant_features}")

# 🎲 Generar variables explicativas (características de cada persona)
# randn genera números aleatorios con distribución normal
X = np.random.randn(n_samples, n_total_features)

# 🎯 Definir coeficientes reales (solo las primeras 10 variables son relevantes)
true_coefficients = np.zeros(n_total_features)  # Inicializar todos en cero
true_coefficients[:n_relevant_features] = np.array([
    5000,  # Educación: Cada año adicional suma $5000 al ingreso
    3000,  # Experiencia laboral: Cada año de experiencia suma $3000
    2000,  # Edad: La edad tiene un efecto moderado
    1500,  # Horas trabajadas: Más horas = más ingreso
    1000,  # Sector económico: Algunos sectores pagan mejor
    800,   # Tamaño de empresa: Empresas grandes suelen pagar más
    600,   # Nivel de responsabilidad: Más responsabilidad = más pago
    400,   # Ubicación geográfica: Algunas ciudades pagan mejor
    300,   # Certificaciones: Certificaciones profesionales aumentan el ingreso
    200    # Idiomas: Cada idioma adicional suma un poco
])

print("\\n💰 Coeficientes verdaderos (solo las primeras 10 variables son relevantes):")
for i, coef in enumerate(true_coefficients[:n_relevant_features]):
    print(f"   Variable {i+1}: ${coef:.0f}")

# 🎯 Generar variable objetivo (ingreso anual) con ruido
# La fórmula es: ingreso = X1*coef1 + X2*coef2 + ... + ruido
y = X @ true_coefficients + np.random.normal(0, 1000, n_samples)

# 📝 Crear nombres de variables para mejor interpretación
feature_names = []
for i in range(n_relevant_features):
    feature_names.append(f'Variable_Relevante_{i+1}')
for i in range(n_irrelevant_features):
    feature_names.append(f'Variable_Irrelevante_{i+1}')

# 📊 Crear DataFrame con pandas
df = pd.DataFrame(X, columns=feature_names)
df['ingreso_anual'] = y

# 📈 Mostrar resumen del dataset
print(f"\\n✅ Dataset creado exitosamente!")
print(f"📊 Observaciones: {n_samples}")
print(f"📈 Variables totales: {n_total_features}")
print(f"💰 Rango de ingresos: ${y.min():.0f} - ${y.max():.0f}")
print(f"💰 Ingreso promedio: ${y.mean():.0f}")
print(f"💰 Desviación estándar: ${y.std():.0f}")

print("\\n📋 Primeras 5 filas del dataset:")
print(df.head())

print("\\n🔍 Información del dataset:")
print(df.info())""")
    
    # === CELDA 5: Preparación de datos ===
    celda5 = nbf.v4.new_markdown_cell("""## 🔧 Paso 2: Preparación de los Datos

### ¿Por qué necesitamos preparar los datos?

Antes de entrenar nuestros modelos, necesitamos hacer algunos ajustes importantes:

1. **Separar variables explicativas y objetivo**: Distinguir entre lo que queremos predecir y lo que usamos para predecir
2. **Dividir en entrenamiento y prueba**: Para evaluar qué tan bien generaliza nuestro modelo
3. **Estandarizar las variables**: Para que todas las variables tengan la misma escala

### 🧠 Conceptos importantes:

**Train-Test Split**: Dividimos nuestros datos en dos partes:
- **Datos de entrenamiento** (70%): Para enseñar al modelo
- **Datos de prueba** (30%): Para evaluar qué tan bien funciona

**Estandarización**: Convertimos todas las variables a la misma escala (media=0, desviación=1). Esto es importante porque:
- Ridge y Lasso son sensibles a la escala de las variables
- Variables con valores grandes pueden dominar el modelo
- La estandarización hace que todas las variables tengan igual importancia inicial""")
    
    # === CELDA 6: Código de preparación ===
    celda6 = nbf.v4.new_code_cell("""# 📊 Separar variables explicativas (X) y variable objetivo (y)
print("🔍 Separando variables explicativas y objetivo...")
X = df.drop('ingreso_anual', axis=1)  # Todas las variables excepto el ingreso
y = df['ingreso_anual']               # Solo el ingreso (lo que queremos predecir)

print(f"📈 Variables explicativas (X): {X.shape[1]} variables")
print(f"🎯 Variable objetivo (y): 1 variable (ingreso anual)")
print(f"📊 Total de observaciones: {X.shape[0]}")

# 🔄 Dividir en conjuntos de entrenamiento y prueba
print("\\n🔄 Dividiendo datos en entrenamiento (70%) y prueba (30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 30% para prueba, semilla para reproducibilidad
)

print(f"📚 Datos de entrenamiento: {X_train.shape[0]} observaciones")
print(f"🧪 Datos de prueba: {X_test.shape[0]} observaciones")
print(f"📈 Variables en cada conjunto: {X_train.shape[1]}")

# ⚖️ Estandarizar las variables (muy importante para Ridge y Lasso)
print("\\n⚖️ Estandarizando variables...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar datos de entrenamiento
X_test_scaled = scaler.transform(X_test)        # Solo transformar datos de prueba

print("✅ Estandarización completada!")
print("📊 Ahora todas las variables tienen media=0 y desviación=1")

# 🔍 Verificar la estandarización
print("\\n🔍 Verificando la estandarización:")
print(f"Media de variables estandarizadas: {X_train_scaled.mean():.6f} (debería ser ~0)")
print(f"Desviación de variables estandarizadas: {X_train_scaled.std():.6f} (debería ser ~1)")

print("\\n✅ ¡Datos preparados y listos para entrenar modelos!")""")
    
    # === CELDA 7: Entrenamiento Ridge ===
    celda7 = nbf.v4.new_markdown_cell("""## 🏔️ Paso 3: Entrenamiento del Modelo Ridge

### ¿Qué es Ridge Regression?

**Ridge Regression** es una técnica de regularización que usa **penalización L2**. Su objetivo es reducir el overfitting agregando una penalización a los coeficientes grandes.

### 🧠 Concepto clave: Penalización L2

La función objetivo de Ridge es:
```
Error = Error de predicción + α × (β₁² + β₂² + ... + βₚ²)
```

Donde:
- **α (alpha)**: Parámetro de regularización (controla la fuerza de la penalización)
- **βᵢ²**: Cuadrado de cada coeficiente

### 🔍 ¿Cómo funciona Ridge?

1. **Reduce coeficientes**: Hace que los coeficientes sean más pequeños
2. **Nunca los hace cero**: Los coeficientes se acercan a cero pero nunca llegan exactamente a cero
3. **Mantiene todas las variables**: Todas las variables siguen en el modelo

### 🎯 ¿Cuándo usar Ridge?

- Cuando todas las variables podrían ser relevantes
- Cuando quieres evitar eliminar variables potencialmente útiles
- Cuando el rendimiento predictivo es la prioridad

### 📊 Proceso que vamos a seguir:

1. **Probar diferentes valores de alpha**: Para encontrar el mejor parámetro
2. **Usar validación cruzada**: Para evaluar cada valor de alpha
3. **Entrenar el modelo final**: Con el mejor alpha encontrado
4. **Evaluar el rendimiento**: En datos de prueba""")
    
    # === CELDA 8: Código Ridge ===
    celda8 = nbf.v4.new_code_cell("""# 🎯 Definir valores de alpha para Ridge
print("🔍 Buscando el mejor parámetro alpha para Ridge...")
print("📊 Probando valores desde 0.001 hasta 1000...")

# Usar logspace para probar valores en escala logarítmica
alpha_values = np.logspace(-3, 3, 50)  # 50 valores entre 10^-3 y 10^3
print(f"🎯 Probando {len(alpha_values)} valores diferentes de alpha")

# 🔄 Entrenar Ridge con validación cruzada para cada alpha
print("\\n🔄 Entrenando modelos Ridge con validación cruzada...")
ridge_scores = []

for i, alpha in enumerate(alpha_values):
    ridge = Ridge(alpha=alpha)
    # Usar validación cruzada con 5 folds
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(-np.mean(scores))  # Convertir a error positivo
    
    # Mostrar progreso cada 10 iteraciones
    if (i + 1) % 10 == 0:
        print(f"   Progreso: {i+1}/{len(alpha_values)} alphas probados")

# 🏆 Encontrar el mejor alpha
best_alpha_ridge = alpha_values[np.argmin(ridge_scores)]
best_score_ridge = min(ridge_scores)

print(f"\\n🏆 Mejor alpha encontrado: {best_alpha_ridge:.4f}")
print(f"📊 Mejor error MSE: {best_score_ridge:.2f}")

# 🚀 Entrenar modelo Ridge final con el mejor alpha
print("\\n🚀 Entrenando modelo Ridge final con el mejor alpha...")
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)

# 📈 Hacer predicciones en datos de prueba
y_pred_ridge = ridge_model.predict(X_test_scaled)

# 📊 Calcular métricas de rendimiento
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\\n📊 RESULTADOS DEL MODELO RIDGE:")
print(f"🎯 Mejor alpha: {best_alpha_ridge:.4f}")
print(f"💰 RMSE (Error de predicción): ${rmse_ridge:.2f}")
print(f"📈 R² (Coeficiente de determinación): {r2_ridge:.4f}")

# 🔍 Analizar coeficientes
ridge_non_zero = np.sum(ridge_model.coef_ != 0)
print(f"📊 Coeficientes no cero: {ridge_non_zero}/{len(ridge_model.coef_)}")
print(f"📊 Porcentaje de variables usadas: {ridge_non_zero/len(ridge_model.coef_)*100:.1f}%")

# 📊 Mostrar algunos coeficientes como ejemplo
print("\\n🔍 Ejemplos de coeficientes Ridge:")
coef_ridge_df = pd.DataFrame({
    'Variable': feature_names,
    'Coeficiente': ridge_model.coef_
})
print(coef_ridge_df.head(10))""")
    
    # === CELDA 9: Entrenamiento Lasso ===
    celda9 = nbf.v4.new_markdown_cell("""## 🎯 Paso 4: Entrenamiento del Modelo Lasso

### ¿Qué es Lasso Regression?

**Lasso Regression** es una técnica de regularización que usa **penalización L1**. Su objetivo es reducir el overfitting y realizar **selección automática de variables**.

### 🧠 Concepto clave: Penalización L1

La función objetivo de Lasso es:
```
Error = Error de predicción + α × (|β₁| + |β₂| + ... + |βₚ|)
```

Donde:
- **α (alpha)**: Parámetro de regularización (controla la fuerza de la penalización)
- **|βᵢ|**: Valor absoluto de cada coeficiente

### 🔍 ¿Cómo funciona Lasso?

1. **Reduce coeficientes**: Hace que los coeficientes sean más pequeños
2. **Puede hacerlos cero**: Los coeficientes pueden llegar exactamente a cero
3. **Selección de variables**: Elimina automáticamente variables irrelevantes

### 🎯 ¿Cuándo usar Lasso?

- Cuando tienes muchas variables y quieres identificar las más importantes
- Cuando la interpretabilidad es crucial
- Cuando quieres un modelo más simple y fácil de explicar
- Cuando sospechas que muchas variables son irrelevantes

### 🆚 Diferencias clave con Ridge:

| Aspecto | Ridge (L2) | Lasso (L1) |
|---------|------------|------------|
| Penalización | βᵢ² (cuadrado) | \|βᵢ\| (valor absoluto) |
| Coeficientes cero | Nunca | Pueden ser cero |
| Selección de variables | No | Sí |
| Interpretabilidad | Baja | Alta |

### 📊 Proceso que vamos a seguir:

1. **Probar diferentes valores de alpha**: Para encontrar el mejor parámetro
2. **Usar validación cruzada**: Para evaluar cada valor de alpha
3. **Entrenar el modelo final**: Con el mejor alpha encontrado
4. **Evaluar el rendimiento**: En datos de prueba
5. **Analizar selección de variables**: Ver qué variables fueron eliminadas""")
    
    # === CELDA 10: Código Lasso ===
    celda10 = nbf.v4.new_code_cell("""# 🎯 Definir valores de alpha para Lasso
print("🔍 Buscando el mejor parámetro alpha para Lasso...")
print("📊 Probando valores desde 0.001 hasta 10...")

# Usar logspace para probar valores en escala logarítmica
alpha_values_lasso = np.logspace(-3, 1, 50)  # 50 valores entre 10^-3 y 10^1
print(f"🎯 Probando {len(alpha_values_lasso)} valores diferentes de alpha")

# 🔄 Entrenar Lasso con validación cruzada para cada alpha
print("\\n🔄 Entrenando modelos Lasso con validación cruzada...")
lasso_scores = []

for i, alpha in enumerate(alpha_values_lasso):
    lasso = Lasso(alpha=alpha, max_iter=2000)  # Más iteraciones para convergencia
    # Usar validación cruzada con 5 folds
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(-np.mean(scores))  # Convertir a error positivo
    
    # Mostrar progreso cada 10 iteraciones
    if (i + 1) % 10 == 0:
        print(f"   Progreso: {i+1}/{len(alpha_values_lasso)} alphas probados")

# 🏆 Encontrar el mejor alpha
best_alpha_lasso = alpha_values_lasso[np.argmin(lasso_scores)]
best_score_lasso = min(lasso_scores)

print(f"\\n🏆 Mejor alpha encontrado: {best_alpha_lasso:.4f}")
print(f"📊 Mejor error MSE: {best_score_lasso:.2f}")

# 🚀 Entrenar modelo Lasso final con el mejor alpha
print("\\n🚀 Entrenando modelo Lasso final con el mejor alpha...")
lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=2000)
lasso_model.fit(X_train_scaled, y_train)

# 📈 Hacer predicciones en datos de prueba
y_pred_lasso = lasso_model.predict(X_test_scaled)

# 📊 Calcular métricas de rendimiento
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\\n📊 RESULTADOS DEL MODELO LASSO:")
print(f"🎯 Mejor alpha: {best_alpha_lasso:.4f}")
print(f"💰 RMSE (Error de predicción): ${rmse_lasso:.2f}")
print(f"📈 R² (Coeficiente de determinación): {r2_lasso:.4f}")

# 🔍 Analizar coeficientes y selección de variables
lasso_non_zero = np.sum(lasso_model.coef_ != 0)
lasso_zero = np.sum(lasso_model.coef_ == 0)

print(f"📊 Coeficientes no cero: {lasso_non_zero}/{len(lasso_model.coef_)}")
print(f"📊 Coeficientes cero (variables eliminadas): {lasso_zero}/{len(lasso_model.coef_)}")
print(f"📊 Porcentaje de variables usadas: {lasso_non_zero/len(lasso_model.coef_)*100:.1f}%")
print(f"📊 Porcentaje de variables eliminadas: {lasso_zero/len(lasso_model.coef_)*100:.1f}%")

# 📊 Mostrar variables seleccionadas y eliminadas
print("\\n🔍 Variables seleccionadas por Lasso (coeficiente ≠ 0):")
lasso_selected_vars = []
lasso_eliminated_vars = []

for i, (var, coef) in enumerate(zip(feature_names, lasso_model.coef_)):
    if coef != 0:
        lasso_selected_vars.append((var, coef))
    else:
        lasso_eliminated_vars.append(var)

print(f"✅ Variables seleccionadas ({len(lasso_selected_vars)}):")
for var, coef in lasso_selected_vars[:10]:  # Mostrar solo las primeras 10
    print(f"   {var}: {coef:.4f}")

if len(lasso_eliminated_vars) > 0:
    print(f"\\n❌ Variables eliminadas ({len(lasso_eliminated_vars)}):")
    for var in lasso_eliminated_vars[:10]:  # Mostrar solo las primeras 10
        print(f"   {var}")

print(f"\\n🎯 ¡Lasso eliminó {len(lasso_eliminated_vars)} variables irrelevantes!")""")
    
    # === CELDA 11: Comparación visual ===
    celda11 = nbf.v4.new_markdown_cell("""## 📊 Paso 5: Comparación Visual de Coeficientes

### ¿Por qué visualizar los coeficientes?

La visualización nos ayuda a entender mejor cómo funcionan Ridge y Lasso:

1. **Ridge**: Todos los coeficientes son pequeños pero no cero
2. **Lasso**: Algunos coeficientes son exactamente cero (variables eliminadas)

### 🧠 Conceptos clave de la visualización:

- **Altura de las barras**: Magnitud del coeficiente
- **Barras en cero**: Variables eliminadas por Lasso
- **Patrón de distribución**: Cómo se distribuyen los coeficientes

### 📈 Lo que vamos a observar:

1. **Diferencias en magnitud**: Ridge vs Lasso
2. **Selección de variables**: Variables eliminadas por Lasso
3. **Interpretabilidad**: Cuál modelo es más fácil de interpretar

### 🎯 Preguntas que responderemos:

- ¿Cuántas variables eliminó Lasso?
- ¿Qué variables considera más importantes cada modelo?
- ¿Cuál modelo es más interpretable?""")
    
    # === CELDA 12: Código de visualización ===
    celda12 = nbf.v4.new_code_cell("""# 📊 Crear figura con subplots para comparar Ridge vs Lasso
print("🎨 Creando visualización comparativa de coeficientes...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

# 🏔️ Gráfico de coeficientes Ridge
print("📈 Preparando gráfico de coeficientes Ridge...")
coef_ridge = pd.Series(ridge_model.coef_, index=feature_names)
coef_ridge_sorted = coef_ridge.sort_values(key=abs, ascending=False)

# Crear barras con colores diferentes para variables relevantes vs irrelevantes
colors_ridge = ['red' if i < n_relevant_features else 'gray' for i in range(len(coef_ridge_sorted))]
coef_ridge_sorted.plot(kind='bar', ax=ax1, color=colors_ridge, alpha=0.7)

ax1.set_title('Coeficientes del Modelo Ridge (L2)', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Valor del Coeficiente', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Agregar leyenda
from matplotlib.patches import Patch
legend_elements_ridge = [
    Patch(facecolor='red', alpha=0.7, label='Variables Relevantes'),
    Patch(facecolor='gray', alpha=0.7, label='Variables Irrelevantes')
]
ax1.legend(handles=legend_elements_ridge, loc='upper right')

# 🎯 Gráfico de coeficientes Lasso
print("📈 Preparando gráfico de coeficientes Lasso...")
coef_lasso = pd.Series(lasso_model.coef_, index=feature_names)
coef_lasso_sorted = coef_lasso.sort_values(key=abs, ascending=False)

# Crear barras con colores diferentes
colors_lasso = ['red' if i < n_relevant_features else 'gray' for i in range(len(coef_lasso_sorted))]
coef_lasso_sorted.plot(kind='bar', ax=ax2, color=colors_lasso, alpha=0.7)

ax2.set_title('Coeficientes del Modelo Lasso (L1)', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Valor del Coeficiente', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Agregar leyenda
legend_elements_lasso = [
    Patch(facecolor='red', alpha=0.7, label='Variables Relevantes'),
    Patch(facecolor='gray', alpha=0.7, label='Variables Irrelevantes')
]
ax2.legend(handles=legend_elements_lasso, loc='upper right')

plt.tight_layout()
plt.show()

# 📊 Análisis detallado de los resultados
print("\\n" + "="*80)
print("📊 ANÁLISIS COMPARATIVO DE COEFICIENTES")
print("="*80)

# 🏔️ Análisis Ridge
print("\\n🏔️ ANÁLISIS RIDGE:")
print(f"📊 Total de variables: {len(coef_ridge)}")
print(f"📊 Variables con coeficiente > 0.1: {np.sum(np.abs(coef_ridge) > 0.1)}")
print(f"📊 Variables con coeficiente > 1.0: {np.sum(np.abs(coef_ridge) > 1.0)}")
print(f"📊 Rango de coeficientes: {coef_ridge.min():.4f} a {coef_ridge.max():.4f}")

# 🎯 Análisis Lasso
print("\\n🎯 ANÁLISIS LASSO:")
print(f"📊 Total de variables: {len(coef_lasso)}")
print(f"📊 Variables seleccionadas (≠ 0): {np.sum(coef_lasso != 0)}")
print(f"📊 Variables eliminadas (= 0): {np.sum(coef_lasso == 0)}")
print(f"📊 Rango de coeficientes: {coef_lasso.min():.4f} a {coef_lasso.max():.4f}")

# 🏆 Comparación
print("\\n🏆 COMPARACIÓN:")
print(f"📊 Variables usadas por Ridge: {len(coef_ridge)} (100%)")
print(f"📊 Variables usadas por Lasso: {np.sum(coef_lasso != 0)} ({np.sum(coef_lasso != 0)/len(coef_lasso)*100:.1f}%)")
print(f"📊 Reducción de variables por Lasso: {len(coef_lasso) - np.sum(coef_lasso != 0)} variables")

# 📋 Mostrar las variables más importantes según cada modelo
print("\\n" + "="*80)
print("📋 TOP 10 VARIABLES MÁS IMPORTANTES")
print("="*80)

print("\\n🏔️ Ridge (por valor absoluto):")
for i, (var, coef) in enumerate(coef_ridge_sorted.head(10).items()):
    relevante = "✅" if i < n_relevant_features else "❌"
    print(f"{i+1:2d}. {var}: {coef:.4f} {relevante}")

print("\\n🎯 Lasso (variables seleccionadas):")
lasso_selected = coef_lasso[coef_lasso != 0].sort_values(key=abs, ascending=False)
for i, (var, coef) in enumerate(lasso_selected.items()):
    relevante = "✅" if var in feature_names[:n_relevant_features] else "❌"
    print(f"{i+1:2d}. {var}: {coef:.4f} {relevante}")

print("\\n📊 Resumen:")
print(f"✅ Ridge identificó {np.sum(np.abs(coef_ridge_sorted.head(10).index.isin(feature_names[:n_relevant_features])))} variables relevantes en su top 10")
print(f"✅ Lasso identificó {np.sum(lasso_selected.head(10).index.isin(feature_names[:n_relevant_features])))} variables relevantes en su top 10")""")
    
    # === CELDA 13: Tabla de resultados ===
    celda13 = nbf.v4.new_markdown_cell("""## 📊 Paso 6: Tabla de Resultados Comparativos

### ¿Por qué crear una tabla comparativa?

Una tabla nos permite ver de manera clara y organizada las diferencias entre Ridge y Lasso en términos de:

1. **Rendimiento predictivo**: ¿Cuál modelo predice mejor?
2. **Interpretabilidad**: ¿Cuál modelo es más fácil de entender?
3. **Selección de variables**: ¿Cuál modelo identifica mejor las variables importantes?

### 🧠 Métricas que vamos a comparar:

- **RMSE**: Error de predicción (menor es mejor)
- **R²**: Coeficiente de determinación (más cercano a 1 es mejor)
- **Número de variables**: Cuántas variables usa cada modelo
- **Capacidad de selección**: Qué tan bien identifica variables relevantes

### 🎯 Preguntas que responderemos:

- ¿Cuál modelo tiene mejor rendimiento predictivo?
- ¿Cuál modelo es más interpretable?
- ¿Cuál modelo identifica mejor las variables relevantes?
- ¿Cuál modelo elimina mejor las variables irrelevantes?""")
    
    # === CELDA 14: Código de tabla ===
    celda14 = nbf.v4.new_code_cell("""# 📊 Crear tabla de resultados comparativos
print("📋 Creando tabla comparativa de resultados...")

# Calcular métricas adicionales para el análisis
ridge_relevant_identified = np.sum(ridge_model.coef_[:n_relevant_features] != 0)
ridge_irrelevant_eliminated = np.sum(ridge_model.coef_[n_relevant_features:] == 0)
lasso_relevant_identified = np.sum(lasso_model.coef_[:n_relevant_features] != 0)
lasso_irrelevant_eliminated = np.sum(lasso_model.coef_[n_relevant_features:] == 0)

# Calcular porcentajes
ridge_relevant_pct = ridge_relevant_identified / n_relevant_features * 100
ridge_irrelevant_pct = ridge_irrelevant_eliminated / n_irrelevant_features * 100
lasso_relevant_pct = lasso_relevant_identified / n_relevant_features * 100
lasso_irrelevant_pct = lasso_irrelevant_eliminated / n_irrelevant_features * 100

# Crear tabla de resultados
resultados = pd.DataFrame({
    'Métrica': [
        '💰 RMSE (Error de Predicción)', 
        '📈 R² (Coeficiente de Determinación)',
        '📊 Número de Variables Usadas',
        '✅ Variables Relevantes Identificadas',
        '❌ Variables Irrelevantes Eliminadas',
        '🎯 Precisión en Selección (%)',
        '🎯 Especificidad (%)'
    ],
    'Ridge': [
        f"${rmse_ridge:.2f}", 
        f"{r2_ridge:.4f}",
        f"{ridge_non_zero}/{len(ridge_model.coef_)} (100%)",
        f"{ridge_relevant_identified}/{n_relevant_features} ({ridge_relevant_pct:.1f}%)",
        f"{ridge_irrelevant_eliminated}/{n_irrelevant_features} ({ridge_irrelevant_pct:.1f}%)",
        f"{ridge_relevant_pct:.1f}%",
        f"{ridge_irrelevant_pct:.1f}%"
    ],
    'Lasso': [
        f"${rmse_lasso:.2f}", 
        f"{r2_lasso:.4f}",
        f"{lasso_non_zero}/{len(lasso_model.coef_)} ({lasso_non_zero/len(lasso_model.coef_)*100:.1f}%)",
        f"{lasso_relevant_identified}/{n_relevant_features} ({lasso_relevant_pct:.1f}%)",
        f"{lasso_irrelevant_eliminated}/{n_irrelevant_features} ({lasso_irrelevant_pct:.1f}%)",
        f"{lasso_relevant_pct:.1f}%",
        f"{lasso_irrelevant_pct:.1f}%"
    ]
})

# Mostrar tabla con formato mejorado
print("\\n" + "="*100)
print("📊 TABLA DE RESULTADOS: RIDGE vs LASSO")
print("="*100)
print(resultados.to_string(index=False))
print("="*100)

# 📊 Análisis de la tabla
print("\\n📊 ANÁLISIS DE RESULTADOS:")

# Comparar rendimiento predictivo
if rmse_lasso < rmse_ridge:
    print(f"🏆 RENDIMIENTO PREDICTIVO: Lasso es mejor por ${rmse_ridge - rmse_lasso:.2f}")
elif rmse_ridge < rmse_lasso:
    print(f"🏆 RENDIMIENTO PREDICTIVO: Ridge es mejor por ${rmse_lasso - rmse_ridge:.2f}")
else:
    print("🏆 RENDIMIENTO PREDICTIVO: Ambos modelos tienen rendimiento similar")

# Comparar interpretabilidad
reduccion_variables = len(ridge_model.coef_) - lasso_non_zero
print(f"\\n📊 INTERPRETABILIDAD:")
print(f"   • Ridge usa todas las {len(ridge_model.coef_)} variables")
print(f"   • Lasso usa solo {lasso_non_zero} variables ({reduccion_variables} menos)")
print(f"   • Lasso eliminó {reduccion_variables/len(ridge_model.coef_)*100:.1f}% de las variables")

# Comparar capacidad de selección
print(f"\\n🎯 CAPACIDAD DE SELECCIÓN:")
print(f"   • Ridge identificó {ridge_relevant_pct:.1f}% de variables relevantes")
print(f"   • Lasso identificó {lasso_relevant_pct:.1f}% de variables relevantes")
print(f"   • Ridge eliminó {ridge_irrelevant_pct:.1f}% de variables irrelevantes")
print(f"   • Lasso eliminó {lasso_irrelevant_pct:.1f}% de variables irrelevantes")

# Determinar el ganador en cada categoría
print(f"\\n🏆 GANADORES POR CATEGORÍA:")
if rmse_lasso <= rmse_ridge:
    print("   🥇 Rendimiento Predictivo: Lasso")
else:
    print("   🥇 Rendimiento Predictivo: Ridge")

if lasso_relevant_pct >= ridge_relevant_pct:
    print("   🥇 Identificación de Variables Relevantes: Lasso")
else:
    print("   🥇 Identificación de Variables Relevantes: Ridge")

if lasso_irrelevant_pct >= ridge_irrelevant_pct:
    print("   🥇 Eliminación de Variables Irrelevantes: Lasso")
else:
    print("   🥇 Eliminación de Variables Irrelevantes: Ridge")

print("   🥇 Interpretabilidad: Lasso (menos variables = más simple)")""")
    
    # === CELDA 15: Análisis detallado ===
    celda15 = nbf.v4.new_markdown_cell("""## Paso 7: Análisis de la Capacidad de Selección de Variables""")
    
    # === CELDA 16: Código de análisis ===
    celda16 = nbf.v4.new_code_cell("""# Análisis detallado de la selección de variables
print("\\n=== ANÁLISIS DE SELECCIÓN DE VARIABLES ===")

# Variables realmente relevantes (primeras 10)
variables_relevantes = feature_names[:n_relevant_features]
variables_irrelevantes = feature_names[n_relevant_features:]

# Análisis Ridge
ridge_relevant_coefs = ridge_model.coef_[:n_relevant_features]
ridge_irrelevant_coefs = ridge_model.coef_[n_relevant_features:]

print(f"\\nRIDGE:")
print(f"- Variables relevantes con coeficiente > 0.1: {np.sum(np.abs(ridge_relevant_coefs) > 0.1)}/{n_relevant_features}")
print(f"- Variables irrelevantes con coeficiente > 0.1: {np.sum(np.abs(ridge_irrelevant_coefs) > 0.1)}/{n_irrelevant_features}")
print(f"- Promedio |coef| variables relevantes: {np.mean(np.abs(ridge_relevant_coefs)):.4f}")
print(f"- Promedio |coef| variables irrelevantes: {np.mean(np.abs(ridge_irrelevant_coefs)):.4f}")

# Análisis Lasso
lasso_relevant_coefs = lasso_model.coef_[:n_relevant_features]
lasso_irrelevant_coefs = lasso_model.coef_[n_relevant_features:]

print(f"\\nLASSO:")
print(f"- Variables relevantes seleccionadas: {np.sum(lasso_relevant_coefs != 0)}/{n_relevant_features}")
print(f"- Variables irrelevantes eliminadas: {np.sum(lasso_irrelevant_coefs == 0)}/{n_irrelevant_features}")
print(f"- Precisión en selección: {np.sum(lasso_relevant_coefs != 0) / n_relevant_features:.2%}")
print(f"- Especificidad: {np.sum(lasso_irrelevant_coefs == 0) / n_irrelevant_features:.2%}")

# Mostrar qué variables relevantes fueron identificadas por Lasso
print(f"\\nVariables relevantes identificadas por Lasso:")
for i, (var, coef) in enumerate(zip(variables_relevantes, lasso_relevant_coefs)):
    status = "✓" if coef != 0 else "✗"
    print(f"{status} {var}: {coef:.4f}")""")
    
    # === CELDA 17: Visualización de evolución ===
    celda17 = nbf.v4.new_markdown_cell("""## Paso 8: Visualización de la Evolución de Coeficientes""")
    
    # === CELDA 18: Código de evolución ===
    celda18 = nbf.v4.new_code_cell("""# Visualizar cómo cambian los coeficientes con diferentes valores de alpha
alphas_ridge = np.logspace(-3, 3, 20)
alphas_lasso = np.logspace(-3, 1, 20)

coefs_ridge = []
coefs_lasso = []

for alpha in alphas_ridge:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_train_scaled, y_train)
    coefs_ridge.append(ridge_temp.coef_)

for alpha in alphas_lasso:
    lasso_temp = Lasso(alpha=alpha, max_iter=2000)
    lasso_temp.fit(X_train_scaled, y_train)
    coefs_lasso.append(lasso_temp.coef_)

coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)

# Crear gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ridge
for i in range(n_total_features):
    color = 'red' if i < n_relevant_features else 'gray'
    alpha_val = 0.8 if i < n_relevant_features else 0.3
    ax1.plot(alphas_ridge, coefs_ridge[:, i], color=color, alpha=alpha_val)
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (Parámetro de Regularización)')
ax1.set_ylabel('Coeficientes')
ax1.set_title('Evolución de Coeficientes - Ridge')
ax1.grid(True, alpha=0.3)

# Lasso
for i in range(n_total_features):
    color = 'red' if i < n_relevant_features else 'gray'
    alpha_val = 0.8 if i < n_relevant_features else 0.3
    ax2.plot(alphas_lasso, coefs_lasso[:, i], color=color, alpha=alpha_val)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (Parámetro de Regularización)')
ax2.set_ylabel('Coeficientes')
ax2.set_title('Evolución de Coeficientes - Lasso')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nObservaciones:")
print("• Líneas rojas: Variables realmente relevantes")
print("• Líneas grises: Variables irrelevantes")
print("• Ridge: Los coeficientes se hacen pequeños pero nunca llegan a cero")
print("• Lasso: Los coeficientes pueden llegar exactamente a cero")""")
    
    # === CELDA 19: Conclusiones ===
    celda19 = nbf.v4.new_markdown_cell("""## Paso 9: Conclusiones y Recomendaciones""")
    
    # === CELDA 20: Código de conclusiones ===
    celda20 = nbf.v4.new_code_cell("""print("\\n" + "="*80)
print("CONCLUSIONES Y RECOMENDACIONES")
print("="*80)

# Comparar rendimiento predictivo
if rmse_lasso < rmse_ridge:
    mejor_prediccion = "Lasso"
    diferencia_rmse = rmse_ridge - rmse_lasso
    print(f"1. RENDIMIENTO PREDICTIVO: Lasso es mejor por ${diferencia_rmse:.2f}")
else:
    mejor_prediccion = "Ridge"
    diferencia_rmse = rmse_lasso - rmse_ridge
    print(f"1. RENDIMIENTO PREDICTIVO: Ridge es mejor por ${diferencia_rmse:.2f}")

# Comparar interpretabilidad
reduccion_variables = len(ridge_model.coef_) - lasso_non_zero
print(f"\\n2. INTERPRETABILIDAD:")
print(f"   • Ridge usa todas las {len(ridge_model.coef_)} variables")
print(f"   • Lasso usa solo {lasso_non_zero} variables ({reduccion_variables} menos)")
print(f"   • Lasso eliminó {reduccion_variables/len(ridge_model.coef_)*100:.1f}% de las variables")

# Análisis de selección correcta
precision_lasso = np.sum(lasso_relevant_coefs != 0) / n_relevant_features
especificidad_lasso = np.sum(lasso_irrelevant_coefs == 0) / n_irrelevant_features

print(f"\\n3. CAPACIDAD DE SELECCIÓN DE VARIABLES:")
print(f"   • Precisión (variables relevantes identificadas): {precision_lasso:.1%}")
print(f"   • Especificidad (variables irrelevantes eliminadas): {especificidad_lasso:.1%}")

# Recomendación final
print(f"\\n4. RECOMENDACIÓN FINAL:")
if precision_lasso > 0.7 and especificidad_lasso > 0.8:
    print("   ✓ Lasso es la mejor opción para este problema")
    print("   • Excelente capacidad de selección de variables")
    print("   • Modelo más interpretable y simple")
elif rmse_lasso < rmse_ridge * 1.05:  # Si Lasso no es más del 5% peor
    print("   ✓ Lasso es recomendable")
    print("   • Rendimiento predictivo similar a Ridge")
    print("   • Ventaja en interpretabilidad")
else:
    print("   ⚠ Ridge podría ser preferible")
    print("   • Mejor rendimiento predictivo")
    print("   • Considerar el trade-off con interpretabilidad")

print("\\n" + "="*80)""")
    
    # === CELDA 21: Resumen final ===
    celda21 = nbf.v4.new_markdown_cell("""## 🎓 Resumen de la Actividad

### 🧠 Lo que hemos aprendido:

#### 1. **Diferencias fundamentales entre Ridge y Lasso:**

| Aspecto | Ridge (L2) | Lasso (L1) |
|---------|------------|------------|
| **Penalización** | βᵢ² (cuadrado) | \|βᵢ\| (valor absoluto) |
| **Coeficientes cero** | Nunca | Pueden ser cero |
| **Selección de variables** | No | Sí |
| **Interpretabilidad** | Baja | Alta |

#### 2. **Capacidad de selección de variables:**
- **Lasso**: Puede identificar automáticamente las variables más importantes
- **Ridge**: Mantiene todas las variables pero con pesos reducidos

#### 3. **Trade-offs importantes:**
- **Interpretabilidad vs. Rendimiento predictivo**
- **Simplicidad del modelo vs. Complejidad**
- **Selección de variables vs. Uso de toda la información**

### 🎯 Aplicaciones prácticas:

#### **Usar Lasso cuando:**
- ✅ Tienes muchas variables y quieres identificar las más importantes
- ✅ La interpretabilidad es crucial
- ✅ Quieres un modelo más simple y fácil de explicar
- ✅ Sospechas que muchas variables son irrelevantes

#### **Usar Ridge cuando:**
- ✅ Todas las variables podrían ser relevantes
- ✅ El rendimiento predictivo es la prioridad máxima
- ✅ Quieres evitar la eliminación de variables potencialmente útiles
- ✅ Tienes correlación alta entre variables

### 🚀 Próximos pasos sugeridos:

1. **📊 Probar con datos reales**: Aplicar estos conceptos al dataset real de ingresos de Perú
2. **🔬 Experimentar con Elastic Net**: Combinación de Ridge y Lasso
3. **🌍 Aplicar a otros problemas**: Usar estos conceptos en otros problemas de regresión
4. **📈 Explorar más técnicas**: Aprender sobre otras técnicas de regularización

### 💡 Conceptos clave para recordar:

- **Regularización**: Técnica para prevenir overfitting
- **Penalización L1 vs L2**: Diferentes formas de regularizar
- **Selección de variables**: Capacidad de eliminar variables irrelevantes
- **Validación cruzada**: Para encontrar el mejor parámetro de regularización
- **Trade-offs**: Siempre hay compensaciones entre diferentes objetivos

### 🎉 ¡Felicidades!

Has completado exitosamente esta actividad práctica sobre Ridge vs Lasso. Ahora tienes una comprensión sólida de:

- ✅ Cómo funcionan las técnicas de regularización
- ✅ Cuándo usar Ridge vs Lasso
- ✅ Cómo interpretar los resultados
- ✅ Cómo evaluar el rendimiento de los modelos

¡Sigue practicando y explorando más técnicas de Machine Learning! 🚀""")
    
    # Agregar todas las celdas al notebook
    nb.cells = [celda1, celda2, celda3, celda4, celda5, celda6, celda7, celda8, 
                celda9, celda10, celda11, celda12, celda13, celda14, celda15, 
                celda16, celda17, celda18, celda19, celda20, celda21]
    
    return nb

if __name__ == "__main__":
    # Crear el notebook
    notebook = crear_notebook_ridge_lasso()
    
    # Guardar el notebook
    nbf.write(notebook, 'Capitulo_4/actividad.ipynb')
    
    print("✅ Notebook creado exitosamente!")
    print("📁 Archivo guardado como: Capitulo_4/actividad.ipynb")
    print("🚀 Puedes abrir el notebook en Jupyter o VS Code") 