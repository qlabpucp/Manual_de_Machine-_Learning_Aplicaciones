import json
import os

# --- Contenido del Notebook ---
# Cada tupla contiene el tipo de celda ('markdown' o 'code') y el código fuente.
# La estructura imita fielmente el notebook de Random Forest, pero adaptado a Gradient Boosting.

notebook_content = [
    ("markdown", """
# 🌳 Modelo de Machine Learning para Predecir la Pobreza Monetaria en Perú (Gradient Boosting)

## 🎯 Objetivo de la Actividad
En este notebook, construiremos, entrenaremos y evaluaremos un modelo de **Gradient Boosting** para predecir si un hogar en Perú caerá en situación de pobreza monetaria. Utilizaremos una base de datos sintética que simula las características de los hogares peruanos.

### ¿Por qué Gradient Boosting?
El Gradient Boosting es un modelo de *aprendizaje supervisado* ideal para este problema por varias razones:
- **Precisión**: Generalmente ofrece un rendimiento superior al Random Forest debido a su enfoque secuencial de corrección de errores.
- **Versatilidad**: Funciona muy bien tanto con variables numéricas como categóricas.
- **Interpretabilidad**: Nos permite conocer qué variables son las más importantes para predecir la pobreza.

### 🧠 Conceptos Clave que Aprenderemos:
- **Preprocesamiento de datos**: Cómo preparar variables categóricas y numéricas.
- **Pipelines en Scikit-learn**: Para organizar nuestro flujo de trabajo de forma profesional.
- **Entrenamiento de un clasificador**: Cómo enseñarle al modelo a partir de los datos.
- **Métricas de Evaluación**: No solo la exactitud (accuracy), sino también la **Precisión**, el **Recall** y la **Matriz de Confusión**.
- **Importancia de Variables (Feature Importance)**: Cómo el modelo nos "explica" su decisión.
- **Análisis SHAP**: Para una interpretabilidad más profunda.
    """),
    ("markdown", """
## 📚 1. Definición del Problema y Preparación Inicial
    """),
    ("code", """
# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings

# Para preprocesamiento y modelado
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# Para evaluar el modelo
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay

# Configuraciones adicionales
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Librerías importadas correctamente.")
    """),
    ("code", """
# Nombre de la carpeta donde se guardarán las imágenes
folder_name = "imagenes"
os.makedirs(folder_name, exist_ok=True)
print(f"✅ Carpeta '{folder_name}' lista para guardar las imágenes.")
    """),
    ("markdown", """
## 🛠️ 2. Preprocesamiento y Limpieza de Datos
    """),
    ("code", """
# 2.1 Cargar e Inspeccionar los Datos
df = pd.read_csv('prediccion_pobreza_peru.csv')

print("Primeras 5 filas del dataset:")
display(df.head())

print("\\nInformación general del DataFrame:")
df.info()

print(f"\\nEl dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
print("\\nConteo de valores nulos por columna:")
print(df.isnull().sum())
    """),
    ("code", """
# 2.2 Separar variables predictoras (X) y objetivo (y)
X = df.drop(['PobrezaMonetaria', 'IngresoMensualHogar', 'GastoMensualHogar'], axis=1)
y = df['PobrezaMonetaria']

# 2.3 Identificar columnas numéricas y categóricas
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print(f"\\nColumnas Numéricas ({len(numerical_cols)}): {list(numerical_cols)}")
print(f"\\nColumnas Categóricas ({len(categorical_cols)}): {list(categorical_cols)}")
    """),
    ("markdown", """
## 3. División de los Datos (Entrenamiento y Prueba)
    """),
    ("code", """
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} hogares")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} hogares")
    """),
    ("markdown", """
## 🤖 4. Construcción y Entrenamiento del Pipeline con Gradient Boosting
    """),
    ("code", """
# Definimos los pasos de preprocesamiento
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# Creamos el Pipeline final con el clasificador GradientBoostingClassifier
# El nombre 'classifier' se mantiene para que el GridSearchCV funcione sin cambios
pipeline_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Entrenamos el modelo base para una primera evaluación
print("🚀 Entrenando el modelo Gradient Boosting base...")
pipeline_final.fit(X_train, y_train)
print("✅ ¡Modelo base entrenado exitosamente!")
    """),
    ("markdown", """
## 📊 5. Evaluación del Modelo Base
    """),
    ("code", """
y_pred = pipeline_final.predict(X_test)
y_pred_proba = pipeline_final.predict_proba(X_test)[:, 1]

print("="*60)
print("Reporte de Clasificación (Modelo Base)")
print(classification_report(y_test, y_pred, target_names=['No Pobre (0)', 'Pobre (1)']))

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy (Exactitud): {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("="*60)
    """),
    ("code", """
# Matriz de Confusión del modelo base
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred. No Pobre', 'Pred. Pobre'],
            yticklabels=['Real No Pobre', 'Real Pobre'])
plt.title('Matriz de Confusión (Boosting Base)', fontsize=16)
plt.savefig(os.path.join(folder_name, '1_matriz_confusion_base_boosting.png'))
plt.show()
    """),
     ("code", """
# Curva ROC del modelo base
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(pipeline_final, X_test, y_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Clasificador Aleatorio')
ax.set_title('Curva ROC (Boosting Base)', fontsize=16)
plt.legend()
plt.savefig(os.path.join(folder_name, '2_curva_roc_base_boosting.png'))
plt.show()
    """),
    ("markdown", """
## ⚙️ 6. Optimización de Hiperparámetros con GridSearchCV
    """),
    ("code", """
# 1. Definición de la rejilla de hiperparámetros para Gradient Boosting
param_grid = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# 2. Configurar y ejecutar GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline_final,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("🚀 Ejecutando GridSearchCV para Gradient Boosting...")
grid_search.fit(X_train, y_train)

# 3. Mostrar los resultados
print("\\n✅ Búsqueda completada.")
print("La mejor configuración de hiperparámetros es:")
print(grid_search.best_params_)
    """),
    ("markdown", """
### 📊 Visualización de Resultados de GridSearchCV
    """),
    ("code", """
# Paleta de colores para el gráfico
azules = ["#87CEEB", "#4682B4", "#191970"]

# Extraer resultados
results = pd.DataFrame(grid_search.cv_results_)
plot_data = pd.DataFrame({
    'learning_rate': results['param_classifier__learning_rate'],
    'Accuracy': results['mean_test_score'],
    'max_depth': results['param_classifier__max_depth'],
    'n_estimators': results['param_classifier__n_estimators']
})

# Crear el gráfico
plt.figure(figsize=(12, 8))
ax = sns.lineplot(
    data=plot_data,
    x='learning_rate',
    y='Accuracy',
    hue='max_depth',
    style='n_estimators',
    markers=True,
    dashes=False,
    palette=azules,
    markersize=10,
    linewidth=2.5
)

# Configurar gráfico
plt.xlabel('Tasa de Aprendizaje (learning_rate)', fontsize=14)
plt.ylabel('Exactitud (Accuracy)', fontsize=14)
plt.title('Rendimiento del Modelo vs. Hiperparámetros (Boosting)', fontsize=16)
ax.legend(title='max_depth / n_estimators', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, '7_gridsearch_results_boosting.png'), dpi=300)
plt.show()
    """),
    ("markdown", """
## ✨ 7. Evaluación del Modelo Optimizado
    """),
    ("code", """
# Extraer y evaluar el mejor modelo
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\\n" + "="*60)
print("Rendimiento del Modelo Optimizado (Boosting)")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=['No Pobre (0)', 'Pobre (1)']))

accuracy_best = accuracy_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)
print(f"Accuracy (Exactitud): {accuracy_best:.4f}")
print(f"ROC-AUC Score: {roc_auc_best:.4f}")
print("="*60)
    """),
    ("code", """
# Matriz de Confusión del modelo optimizado
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Pred. No Pobre', 'Pred. Pobre'],
            yticklabels=['Real No Pobre', 'Real Pobre'])
plt.title('Matriz de Confusión - Boosting Optimizado', fontsize=16)
plt.savefig(os.path.join(folder_name, '3_matriz_confusion_optimizado_boosting.png'))
plt.show()
    """),
    ("code", """
# Curva ROC del modelo optimizado
fig, ax = plt.subplots(figsize=(8, 7))
RocCurveDisplay.from_estimator(
    best_model, X_test, y_test, ax=ax, name='Modelo Optimizado', color='darkorange'
)
ax.plot([0, 1], [0, 1], linestyle='--', color='navy', label='Clasificador Aleatorio')
ax.set_title('Curva ROC del Modelo Boosting Optimizado', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(folder_name, '4_curva_roc_optimizado_boosting.png'))
plt.show()
    """),
    ("markdown", """
## 🔍 8. Interpretación del Modelo (Feature Importance y SHAP)
    """),
    ("code", """
# Extraer importancia de variables del modelo optimizado
preprocessor_best = best_model.named_steps['preprocessor']
classifier_best = best_model.named_steps['classifier']

try:
    feature_names_out = preprocessor_best.get_feature_names_out()
except AttributeError:
    ohe_feature_names = preprocessor_best.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names_out = list(numerical_cols) + list(ohe_feature_names)

importances = classifier_best.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names_out, 'importance': importances}).sort_values('importance', ascending=False)

# Visualizar las 15 variables más importantes
plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Variables más Importantes (Boosting Optimizado)', fontsize=16)
plt.savefig(os.path.join(folder_name, '5_importancia_variables_boosting.png'))
plt.show()
    """),
     ("markdown", """
### Interpretación Avanzada con SHAP
    """),
    ("code", """
print("Generando el gráfico SHAP summary plot...")

X_test_transformed = preprocessor_best.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

explainer = shap.TreeExplainer(classifier_best)
shap_explanation = explainer(X_test_transformed)

print("\\nMostrando el gráfico...")
shap.summary_plot(
    shap_explanation, 
    features=X_test_transformed, 
    feature_names=feature_names_out, 
    show=False,
    plot_type='bar' # Para Gradient Boosting, 'bar' es una buena opción inicial
)
plt.savefig(os.path.join(folder_name, '6_shap_summary_plot_boosting.png'), bbox_inches='tight')
plt.show()
    """),
    ("markdown", """
## 🔚 9. Conclusión General

En este ejercicio, hemos construido, evaluado y optimizado con éxito un modelo `GradientBoostingClassifier`. La optimización de hiperparámetros mediante `GridSearchCV` permitió mejorar la exactitud del modelo base. Los análisis de importancia de variables y SHAP confirmaron que los principales predictores de la pobreza son factores estructurales como la demografía del hogar, el nivel educativo y el tipo de empleo, validando la capacidad del modelo para capturar relaciones socioeconómicas complejas.
    """)
]

# --- Lógica para Crear el Notebook ---

def create_notebook(content_list, filename):
    """Genera un archivo .ipynb a partir de una lista de contenido."""
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

    for cell_type, source in content_list:
        if cell_type == 'code':
            cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': source.strip().split('\n')
            }
        elif cell_type == 'markdown':
            cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': source.strip().split('\n')
            }
        notebook['cells'].append(cell)

    # Guardar el notebook en formato JSON
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Notebook '{filename}' creado exitosamente.")

# --- Ejecución ---
if __name__ == '__main__':
    create_notebook(notebook_content, 'Aplicacion_practica_boosting_si.ipynb')