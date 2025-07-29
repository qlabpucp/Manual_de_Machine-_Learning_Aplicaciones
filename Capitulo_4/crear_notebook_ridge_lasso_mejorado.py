import nbformat as nbf

def crear_notebook_ridge_lasso_mejorado():
    """Crear un notebook mejorado para evidenciar claramente las diferencias entre Ridge y Lasso"""
    
    # Crear un nuevo notebook
    nb = nbf.v4.new_notebook()
    
    # === CELDA 1: Título principal ===
    celda1 = nbf.v4.new_markdown_cell("""# Actividad Mejorada: Lasso vs. Ridge - Evidenciando la Selección de Variables

## 🎯 Objetivo de la Actividad
En esta actividad práctica, vamos a crear un experimento diseñado específicamente para **evidenciar claramente** las diferencias entre Ridge y Lasso, especialmente la capacidad de Lasso para realizar selección automática de variables.

## 🧪 Diseño del Experimento

### ¿Por qué este experimento es diferente?
- **Más variables irrelevantes**: 50 variables irrelevantes vs 5 relevantes
- **Coeficientes más extremos**: Variables relevantes con coeficientes muy altos
- **Variables irrelevantes con coeficiente cero**: Para que Lasso las elimine completamente
- **Menos ruido**: Para que las diferencias sean más claras

### 📊 Estructura de datos:
- **5 variables relevantes**: Con coeficientes muy altos (10000, 8000, 6000, 4000, 2000)
- **50 variables irrelevantes**: Con coeficiente real = 0
- **1000 observaciones**: Para tener suficientes datos
- **Ruido mínimo**: Para evidenciar las diferencias

### 🎯 Hipótesis específicas:
1. **Lasso** eliminará la mayoría de las 50 variables irrelevantes
2. **Ridge** mantendrá todas las variables pero con coeficientes pequeños
3. **Lasso** será mucho más interpretable (5-10 variables vs 55 variables)
4. **Ridge** tendrá mejor rendimiento predictivo pero será menos interpretable""")
    
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
print("🚀 ¡Listos para comenzar la actividad mejorada!")""")
    
    # === CELDA 3: Generación de datos mejorados ===
    celda3 = nbf.v4.new_markdown_cell("""## 🧪 Paso 1: Generación de Datos Optimizados

### 🎯 Diseño específico para evidenciar diferencias:

**Variables relevantes (5):**
1. **Educación** (coef = 10000): Variable muy importante
2. **Experiencia** (coef = 8000): Variable importante
3. **Edad** (coef = 6000): Variable moderadamente importante
4. **Horas** (coef = 4000): Variable menos importante
5. **Sector** (coef = 2000): Variable poco importante

**Variables irrelevantes (50):**
- Todas con coeficiente real = 0
- Generadas aleatoriamente
- Sin relación con el ingreso

### 🔍 ¿Por qué este diseño?
- **Coeficientes extremos**: Para que Lasso identifique claramente las variables importantes
- **Muchas variables irrelevantes**: Para que la selección sea dramática
- **Sin correlaciones**: Para evitar confusión""")
    
    # === CELDA 4: Código de generación mejorada ===
    celda4 = nbf.v4.new_code_cell("""# 🔧 Configurar semilla para reproducibilidad
# Esto asegura que obtengamos los mismos resultados cada vez que ejecutemos el código
np.random.seed(42)

# 📊 Parámetros optimizados para evidenciar diferencias
n_samples = 1000                    # Número de personas en nuestro dataset
n_relevant_features = 5             # Variables que realmente afectan el ingreso
n_irrelevant_features = 50          # Variables que NO afectan el ingreso
n_total_features = n_relevant_features + n_irrelevant_features

print(f"🎯 Creando dataset optimizado:")
print(f"📈 Variables relevantes: {n_relevant_features}")
print(f"❌ Variables irrelevantes: {n_irrelevant_features}")
print(f"📊 Total de variables: {n_total_features}")

# 🎲 Generar variables explicativas (características de cada persona)
# randn genera números aleatorios con distribución normal
X = np.random.randn(n_samples, n_total_features)

# 🎯 Definir coeficientes reales (solo las primeras 5 variables son relevantes)
true_coefficients = np.zeros(n_total_features)  # Inicializar todos en cero
true_coefficients[:n_relevant_features] = np.array([
    10000,  # Educación: Variable muy importante - cada año suma $10000
    8000,   # Experiencia: Variable importante - cada año suma $8000
    6000,   # Edad: Variable moderadamente importante - cada año suma $6000
    4000,   # Horas: Variable menos importante - cada hora suma $4000
    2000    # Sector: Variable poco importante - cada nivel suma $2000
])

print("\\n💰 Coeficientes verdaderos (solo las primeras 5 variables son relevantes):")
for i, coef in enumerate(true_coefficients[:n_relevant_features]):
    print(f"   Variable {i+1}: ${coef:.0f}")

# 🎯 Generar variable objetivo (ingreso anual) con ruido mínimo
# La fórmula es: ingreso = X1*coef1 + X2*coef2 + ... + ruido
y = X @ true_coefficients + np.random.normal(0, 500, n_samples)  # Menos ruido para evidenciar diferencias

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
print(f"\\n✅ Dataset optimizado creado exitosamente!")
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

### 🎯 Preparación estándar:
1. **Separar variables explicativas y objetivo**
2. **Dividir en entrenamiento y prueba**
3. **Estandarizar las variables**

### 🧠 ¿Por qué es crucial la estandarización?
- Ridge y Lasso son muy sensibles a la escala
- Sin estandarización, las variables con valores grandes dominarían
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

### 🧠 ¿Qué esperamos de Ridge?
- **Mantendrá todas las 55 variables**
- **Reducirá los coeficientes pero nunca los hará cero**
- **Mejor rendimiento predictivo** (usa toda la información)
- **Menos interpretable** (55 variables vs pocas)

### 📊 Proceso:
1. **Probar diferentes valores de alpha**
2. **Usar validación cruzada**
3. **Entrenar modelo final**
4. **Analizar coeficientes**""")
    
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

### 🧠 ¿Qué esperamos de Lasso?
- **Eliminará la mayoría de las 50 variables irrelevantes**
- **Mantendrá las 5 variables relevantes**
- **Coeficientes exactamente cero** para variables eliminadas
- **Modelo mucho más interpretable** (5-10 variables vs 55)

### 📊 Proceso:
1. **Probar diferentes valores de alpha**
2. **Usar validación cruzada**
3. **Entrenar modelo final**
4. **Analizar selección de variables**""")
    
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
    
    # === CELDA 11: Comparación visual mejorada ===
    celda11 = nbf.v4.new_markdown_cell("""## 📊 Paso 5: Comparación Visual Mejorada

### 🎯 Visualización diseñada para evidenciar diferencias:

1. **Gráfico de coeficientes**: Mostrar claramente cómo Ridge mantiene todas las variables
2. **Gráfico de Lasso**: Mostrar cómo Lasso elimina variables irrelevantes
3. **Análisis detallado**: Contar variables seleccionadas vs eliminadas
4. **Comparación de rendimiento**: RMSE y R² de ambos modelos

### 🧠 Lo que vamos a observar:
- **Ridge**: 55 barras pequeñas (todas las variables)
- **Lasso**: Solo 5-10 barras (variables seleccionadas)
- **Diferencia dramática** en interpretabilidad""")
    
    # === CELDA 12: Código de visualización mejorada ===
    celda12 = nbf.v4.new_code_cell("""# 📊 Crear figura con subplots para comparar Ridge vs Lasso
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

# 🏔️ Gráfico de coeficientes Ridge
coef_ridge = pd.Series(ridge_model.coef_, index=feature_names)
coef_ridge_sorted = coef_ridge.sort_values(key=abs, ascending=False)

# Crear barras con colores diferentes
colors_ridge = ['red' if i < n_relevant_features else 'gray' for i in range(len(coef_ridge_sorted))]
coef_ridge_sorted.plot(kind='bar', ax=ax1, color=colors_ridge, alpha=0.7)

ax1.set_title('Coeficientes del Modelo Ridge (L2) - TODAS LAS VARIABLES', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Valor del Coeficiente', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Agregar leyenda
from matplotlib.patches import Patch
legend_elements_ridge = [
    Patch(facecolor='red', alpha=0.7, label='Variables Relevantes (5)'),
    Patch(facecolor='gray', alpha=0.7, label='Variables Irrelevantes (50)')
]
ax1.legend(handles=legend_elements_ridge, loc='upper right')

# 🎯 Gráfico de coeficientes Lasso
coef_lasso = pd.Series(lasso_model.coef_, index=feature_names)
coef_lasso_sorted = coef_lasso.sort_values(key=abs, ascending=False)

# Crear barras con colores diferentes
colors_lasso = ['red' if i < n_relevant_features else 'gray' for i in range(len(coef_lasso_sorted))]
coef_lasso_sorted.plot(kind='bar', ax=ax2, color=colors_lasso, alpha=0.7)

ax2.set_title('Coeficientes del Modelo Lasso (L1) - SOLO VARIABLES SELECCIONADAS', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('Valor del Coeficiente', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Agregar leyenda
legend_elements_lasso = [
    Patch(facecolor='red', alpha=0.7, label='Variables Relevantes (5)'),
    Patch(facecolor='gray', alpha=0.7, label='Variables Irrelevantes (50)')
]
ax2.legend(handles=legend_elements_lasso, loc='upper right')

plt.tight_layout()
plt.show()

# 📊 Análisis detallado
print("\\n" + "="*80)
print("📊 ANÁLISIS COMPARATIVO MEJORADO")
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

# 🏆 Comparación dramática
print("\\n🏆 COMPARACIÓN DRAMÁTICA:")
print(f"📊 Variables usadas por Ridge: {len(coef_ridge)} (100%)")
print(f"📊 Variables usadas por Lasso: {np.sum(coef_lasso != 0)} ({np.sum(coef_lasso != 0)/len(coef_lasso)*100:.1f}%)")
print(f"📊 Reducción de variables por Lasso: {len(coef_lasso) - np.sum(coef_lasso != 0)} variables")

# 📋 Mostrar las variables más importantes
print("\\n" + "="*80)
print("📋 VARIABLES MÁS IMPORTANTES")
print("="*80)

print("\\n🏔️ Ridge (top 10 por valor absoluto):")
for i, (var, coef) in enumerate(coef_ridge_sorted.head(10).items()):
    relevante = "✅" if var in feature_names[:n_relevant_features] else "❌"
    print(f"{i+1:2d}. {var}: {coef:.4f} {relevante}")

print("\\n🎯 Lasso (variables seleccionadas):")
lasso_selected = coef_lasso[coef_lasso != 0].sort_values(key=abs, ascending=False)
for i, (var, coef) in enumerate(lasso_selected.items()):
    relevante = "✅" if var in feature_names[:n_relevant_features] else "❌"
    print(f"{i+1:2d}. {var}: {coef:.4f} {relevante}")

print("\\n📊 Resumen:")
print(f"✅ Ridge identificó {np.sum(np.abs(coef_ridge_sorted.head(10).index.isin(feature_names[:n_relevant_features])))} variables relevantes en su top 10")
print(f"✅ Lasso identificó {np.sum(lasso_selected.head(10).index.isin(feature_names[:n_relevant_features]))} variables relevantes en su top 10")""")
    
    # === CELDA 13: Tabla de resultados mejorada ===
    celda13 = nbf.v4.new_markdown_cell("""## 📊 Paso 6: Tabla de Resultados Mejorada

### 🎯 Tabla diseñada para evidenciar diferencias:

Vamos a crear una tabla que muestre claramente:
1. **Rendimiento predictivo**: RMSE y R²
2. **Interpretabilidad**: Número de variables usadas
3. **Capacidad de selección**: Variables relevantes identificadas
4. **Eliminación de variables**: Variables irrelevantes eliminadas

### 🧠 Métricas clave:
- **RMSE**: Error de predicción
- **R²**: Coeficiente de determinación
- **Variables usadas**: Cuántas variables usa cada modelo
- **Precisión**: Qué tan bien identifica variables relevantes
- **Especificidad**: Qué tan bien elimina variables irrelevantes""")
    
    # === CELDA 14: Código de tabla mejorada ===
    celda14 = nbf.v4.new_code_cell("""# 📊 Crear tabla de resultados comparativos
print("📋 Creando tabla comparativa mejorada...")

# Calcular métricas específicas
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

# Mostrar tabla
print("\\n" + "="*100)
print("📊 TABLA DE RESULTADOS MEJORADA: RIDGE vs LASSO")
print("="*100)
print(resultados.to_string(index=False))
print("="*100)

# 📊 Análisis dramático
print("\\n📊 ANÁLISIS DRAMÁTICO DE RESULTADOS:")

# Comparar rendimiento predictivo
if rmse_lasso < rmse_ridge:
    print(f"🏆 RENDIMIENTO PREDICTIVO: Lasso es mejor por ${rmse_ridge - rmse_lasso:.2f}")
elif rmse_ridge < rmse_lasso:
    print(f"🏆 RENDIMIENTO PREDICTIVO: Ridge es mejor por ${rmse_lasso - rmse_ridge:.2f}")
else:
    print("🏆 RENDIMIENTO PREDICTIVO: Ambos modelos tienen rendimiento similar")

# Comparar interpretabilidad (dramática)
reduccion_variables = len(ridge_model.coef_) - lasso_non_zero
print(f"\\n📊 INTERPRETABILIDAD (DIFERENCIA DRAMÁTICA):")
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

print("   🥇 Interpretabilidad: Lasso (dramáticamente más simple)")""")
    
    # === CELDA 15: Conclusiones mejoradas ===
    celda15 = nbf.v4.new_markdown_cell("""## 🎓 Paso 7: Conclusiones Mejoradas

### 🧠 Lo que hemos evidenciado claramente:

#### 1. **Diferencias dramáticas en interpretabilidad:**
- **Ridge**: Usa todas las 55 variables (100%)
- **Lasso**: Usa solo 5-10 variables (9-18%)
- **Reducción**: Lasso eliminó 45-50 variables (82-91%)

#### 2. **Capacidad de selección de variables:**
- **Lasso**: Eliminó la mayoría de las variables irrelevantes
- **Ridge**: Mantuvo todas las variables pero con coeficientes pequeños
- **Precisión**: Lasso identificó correctamente las variables relevantes

#### 3. **Trade-offs claros:**
- **Ridge**: Mejor rendimiento predictivo, menos interpretable
- **Lasso**: Rendimiento similar, mucho más interpretable
- **Selección**: Lasso es superior para identificar variables importantes

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

1. **📊 Probar con datos reales**: Aplicar estos conceptos a datasets reales
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

Has completado exitosamente esta actividad práctica mejorada sobre Ridge vs Lasso. Ahora tienes una comprensión sólida de:

- ✅ Cómo funcionan las técnicas de regularización
- ✅ Cuándo usar Ridge vs Lasso
- ✅ Cómo interpretar los resultados
- ✅ Cómo evaluar el rendimiento de los modelos
- ✅ La capacidad dramática de Lasso para seleccionar variables

¡Sigue practicando y explorando más técnicas de Machine Learning! 🚀""")
    
    # Agregar todas las celdas al notebook
    nb.cells = [celda1, celda2, celda3, celda4, celda5, celda6, celda7, celda8, 
                celda9, celda10, celda11, celda12, celda13, celda14, celda15]
    
    return nb

if __name__ == "__main__":
    # Crear el notebook
    notebook = crear_notebook_ridge_lasso_mejorado()
    
    # Guardar el notebook
    nbf.write(notebook, 'Capitulo_4/actividad_mejorada.ipynb')
    
    print("✅ Notebook mejorado creado exitosamente!")
    print("📁 Archivo guardado como: Capitulo_4/actividad_mejorada.ipynb")
    print("🚀 Puedes abrir el notebook en Jupyter o VS Code")
    print("🎯 Este notebook está diseñado para evidenciar claramente las diferencias entre Ridge y Lasso") 