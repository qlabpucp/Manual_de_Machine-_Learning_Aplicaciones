
import nbformat as nbf

def crear_notebook_ridge_lasso():
    """Crear un notebook completo para comparar Ridge vs Lasso"""
    
    # Crear un nuevo notebook
    nb = nbf.v4.new_notebook()
    
    # Lista de todas las celdas
    celdas = []
    
    # === CELDA 1: Título principal ===
    celda1 = nbf.v4.new_markdown_cell("""# Actividad: Lasso vs. Ridge - Comparación de Modelos de Regularización

## Objetivo
Comparar empíricamente el rendimiento y la interpretabilidad de las regresiones Lasso y Ridge usando datos simulados que incluyen variables relevantes e irrelevantes para demostrar la capacidad de Lasso de realizar selección automática de características.

## Hipótesis
1. Lasso tendrá un error de predicción similar o mejor que Ridge
2. Lasso producirá un modelo más interpretable al reducir a cero los coeficientes de variables irrelevantes
3. Ridge mantendrá todos los coeficientes pero con valores pequeños""")
    
    # === CELDA 2: Importaciones ===
    celda2 = nbf.v4.new_code_cell("""# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Librerías importadas correctamente")""")
    
    # === CELDA 3: Sección de datos simulados ===
    celda3 = nbf.v4.new_markdown_cell("""## Paso 1: Generación de Datos Simulados

Crearemos un conjunto de datos que incluya:
- Variables relevantes que realmente afectan el ingreso
- Variables irrelevantes que no tienen relación con el ingreso
- Ruido para simular condiciones reales""")
    
    # === CELDA 4: Código de generación de datos ===
    celda4 = nbf.v4.new_code_cell("""# Configurar semilla para reproducibilidad
np.random.seed(42)

# Parámetros de la simulación
n_samples = 1000
n_relevant_features = 10  # Variables que realmente afectan el ingreso
n_irrelevant_features = 20  # Variables que no afectan el ingreso
n_total_features = n_relevant_features + n_irrelevant_features

# Generar variables explicativas
X = np.random.randn(n_samples, n_total_features)

# Definir coeficientes reales (solo las primeras 10 variables son relevantes)
true_coefficients = np.zeros(n_total_features)
true_coefficients[:n_relevant_features] = np.array([
    5000,  # Educación
    3000,  # Experiencia laboral
    2000,  # Edad
    1500,  # Horas trabajadas
    1000,  # Sector económico
    800,   # Tamaño de empresa
    600,   # Nivel de responsabilidad
    400,   # Ubicación geográfica
    300,   # Certificaciones
    200    # Idiomas
])

# Generar variable objetivo con ruido
y = X @ true_coefficients + np.random.normal(0, 1000, n_samples)

# Crear DataFrame con nombres de variables
feature_names = []
for i in range(n_relevant_features):
    feature_names.append(f'Variable_Relevante_{i+1}')
for i in range(n_irrelevant_features):
    feature_names.append(f'Variable_Irrelevante_{i+1}')

df = pd.DataFrame(X, columns=feature_names)
df['ingreso_anual'] = y

print(f"Dataset creado con {n_samples} observaciones y {n_total_features} variables")
print(f"Variables relevantes: {n_relevant_features}")
print(f"Variables irrelevantes: {n_irrelevant_features}")
print(f"Rango de ingresos: ${y.min():.0f} - ${y.max():.0f}")
print("\\nPrimeras 5 filas del dataset:")
print(df.head())""")
    
    # === CELDA 5: Preparación de datos ===
    celda5 = nbf.v4.new_markdown_cell("""## Paso 2: Preparación de los Datos""")
    
    # === CELDA 6: Código de preparación ===
    celda6 = nbf.v4.new_code_cell("""# Separar variables explicativas y objetivo
X = df.drop('ingreso_anual', axis=1)
y = df['ingreso_anual']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Estandarizar las variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} observaciones")
print(f"Conjunto de prueba: {X_test.shape[0]} observaciones")
print(f"Número de variables: {X_train.shape[1]}")""")
    
    # === CELDA 7: Entrenamiento Ridge ===
    celda7 = nbf.v4.new_markdown_cell("""## Paso 3: Entrenamiento del Modelo Ridge""")
    
    # === CELDA 8: Código Ridge ===
    celda8 = nbf.v4.new_code_cell("""# Definir valores de alpha para Ridge
alpha_values = np.logspace(-3, 3, 50)

# Entrenar Ridge con validación cruzada
ridge_scores = []
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(-np.mean(scores))

# Encontrar el mejor alpha
best_alpha_ridge = alpha_values[np.argmin(ridge_scores)]

# Entrenar modelo Ridge final con el mejor alpha
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)

# Predicciones
y_pred_ridge = ridge_model.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Mejor alpha para Ridge: {best_alpha_ridge:.4f}")
print(f"RMSE Ridge: ${rmse_ridge:.2f}")
print(f"R² Ridge: {r2_ridge:.4f}")

# Contar coeficientes no cero
ridge_non_zero = np.sum(ridge_model.coef_ != 0)
print(f"Coeficientes no cero en Ridge: {ridge_non_zero}/{len(ridge_model.coef_)}")""")
    
    # === CELDA 9: Entrenamiento Lasso ===
    celda9 = nbf.v4.new_markdown_cell("""## Paso 4: Entrenamiento del Modelo Lasso""")
    
    # === CELDA 10: Código Lasso ===
    celda10 = nbf.v4.new_code_cell("""# Definir valores de alpha para Lasso
alpha_values_lasso = np.logspace(-3, 1, 50)

# Entrenar Lasso con validación cruzada
lasso_scores = []
for alpha in alpha_values_lasso:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(-np.mean(scores))

# Encontrar el mejor alpha
best_alpha_lasso = alpha_values_lasso[np.argmin(lasso_scores)]

# Entrenar modelo Lasso final con el mejor alpha
lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=2000)
lasso_model.fit(X_train_scaled, y_train)

# Predicciones
y_pred_lasso = lasso_model.predict(X_test_scaled)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Mejor alpha para Lasso: {best_alpha_lasso:.4f}")
print(f"RMSE Lasso: ${rmse_lasso:.2f}")
print(f"R² Lasso: {r2_lasso:.4f}")

# Contar coeficientes no cero
lasso_non_zero = np.sum(lasso_model.coef_ != 0)
print(f"Coeficientes no cero en Lasso: {lasso_non_zero}/{len(lasso_model.coef_)}")
print(f"Variables seleccionadas por Lasso: {lasso_non_zero}")""")
    
    # === CELDA 11: Comparación visual ===
    celda11 = nbf.v4.new_markdown_cell("""## Paso 5: Comparación Visual de Coeficientes""")
    
    # === CELDA 12: Código de visualización ===
    celda12 = nbf.v4.new_code_cell("""# Crear figura con subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Gráfico de coeficientes Ridge
coef_ridge = pd.Series(ridge_model.coef_, index=feature_names)
coef_ridge_sorted = coef_ridge.sort_values(key=abs, ascending=False)
coef_ridge_sorted.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
ax1.set_title('Coeficientes del Modelo Ridge', fontsize=14, fontweight='bold')
ax1.set_ylabel('Valor del Coeficiente')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Gráfico de coeficientes Lasso
coef_lasso = pd.Series(lasso_model.coef_, index=feature_names)
coef_lasso_sorted = coef_lasso.sort_values(key=abs, ascending=False)
coef_lasso_sorted.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.7)
ax2.set_title('Coeficientes del Modelo Lasso', fontsize=14, fontweight='bold')
ax2.set_ylabel('Valor del Coeficiente')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mostrar las variables más importantes según cada modelo
print("\\n=== TOP 10 VARIABLES MÁS IMPORTANTES ===")
print("\\nRidge (por valor absoluto):")
for i, (var, coef) in enumerate(coef_ridge_sorted.head(10).items()):
    print(f"{i+1:2d}. {var}: {coef:.2f}")

print("\\nLasso (variables seleccionadas):")
lasso_selected = coef_lasso[coef_lasso != 0].sort_values(key=abs, ascending=False)
for i, (var, coef) in enumerate(lasso_selected.items()):
    print(f"{i+1:2d}. {var}: {coef:.2f}")""")
    
    # === CELDA 13: Tabla de resultados ===
    celda13 = nbf.v4.new_markdown_cell("""## Paso 6: Tabla de Resultados Comparativos""")
    
    # === CELDA 14: Código de tabla ===
    celda14 = nbf.v4.new_code_cell("""# Crear tabla de resultados
resultados = pd.DataFrame({
    'Métrica': ['RMSE (Error de Predicción)', 
                'R² (Coeficiente de Determinación)',
                'Número de Variables Usadas',
                'Variables Relevantes Identificadas',
                'Variables Irrelevantes Eliminadas'],
    'Ridge': [f"${rmse_ridge:.2f}", 
              f"{r2_ridge:.4f}",
              f"{ridge_non_zero}/{len(ridge_model.coef_)}",
              f"{np.sum(ridge_model.coef_[:n_relevant_features] != 0)}/{n_relevant_features}",
              f"{np.sum(ridge_model.coef_[n_relevant_features:] == 0)}/{n_irrelevant_features}"],
    'Lasso': [f"${rmse_lasso:.2f}", 
              f"{r2_lasso:.4f}",
              f"{lasso_non_zero}/{len(lasso_model.coef_)}",
              f"{np.sum(lasso_model.coef_[:n_relevant_features] != 0)}/{n_relevant_features}",
              f"{np.sum(lasso_model.coef_[n_relevant_features:] == 0)}/{n_irrelevant_features}"]
})

print("\\n" + "="*80)
print("TABLA DE RESULTADOS: RIDGE vs LASSO")
print("="*80)
print(resultados.to_string(index=False))
print("="*80)""")
    
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
    celda21 = nbf.v4.new_markdown_cell("""## Resumen de la Actividad

### Lo que hemos aprendido:

1. **Diferencias fundamentales entre Ridge y Lasso:**
   - Ridge: Penalización L2, coeficientes pequeños pero no cero
   - Lasso: Penalización L1, coeficientes pueden llegar exactamente a cero

2. **Capacidad de selección de variables:**
   - Lasso puede identificar automáticamente las variables más importantes
   - Ridge mantiene todas las variables pero con pesos reducidos

3. **Trade-offs:**
   - Interpretabilidad vs. Rendimiento predictivo
   - Simplicidad del modelo vs. Complejidad

### Aplicaciones prácticas:

**Usar Lasso cuando:**
- Tienes muchas variables y quieres identificar las más importantes
- La interpretabilidad es crucial
- Quieres un modelo más simple y fácil de explicar

**Usar Ridge cuando:**
- Todas las variables podrían ser relevantes
- El rendimiento predictivo es la prioridad máxima
- Quieres evitar la eliminación de variables potencialmente útiles

### Próximos pasos:
1. Probar con el dataset real de ingresos de Perú
2. Experimentar con Elastic Net (combinación de Ridge y Lasso)
3. Aplicar estos conceptos a otros problemas de regresión""")
    
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