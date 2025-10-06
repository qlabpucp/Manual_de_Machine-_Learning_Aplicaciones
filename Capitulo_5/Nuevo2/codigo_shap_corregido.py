# Código SHAP corregido para el notebook Aplicacion_practica_revisado.ipynb

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Instalar SHAP si no está instalado
try:
    import shap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# Verificar si las variables necesarias están definidas, si no, ejecutar el código necesario
try:
    best_model
    X_test
    all_feature_names
    print("✅ Variables necesarias para SHAP encontradas")
except NameError as e:
    print(f"⚠️ Error: {e}")
    print("Ejecutando código necesario para definir las variables...")
    
    # Cargar datos si no están disponibles
    if 'df' not in locals():
        df = pd.read_csv('prediccion_pobreza_peru.csv')
        X = df.drop(['PobrezaMonetaria', 'IngresoMensualHogar', 'GastoMensualHogar'], axis=1)
        y = df['PobrezaMonetaria']
        
        # Identificar columnas numéricas y categóricas
        numerical_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Dividir datos
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Crear pipeline
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numerical_cols),
                         ('cat', categorical_transformer, categorical_cols)],
            remainder='passthrough'
        )
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        pipeline_final = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
        
        # Entrenar modelo
        pipeline_final.fit(X_train, y_train)
        
        # Optimizar con GridSearchCV si best_model no está definido
        if 'best_model' not in locals():
            param_grid = {
                'classifier__n_estimators': [150, 250, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(
                estimator=pipeline_final,
                param_grid=param_grid,
                scoring='accuracy',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        
        # Obtener nombres de características
        try:
            ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        except AttributeError:
            ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names(categorical_cols)
        
        all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])

# Crear el explicador SHAP con el clasificador (no el pipeline completo)
print("🔍 Creando explicador SHAP...")
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

# Preprocesar los datos de prueba
print("📊 Preprocesando datos de prueba...")
X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)

# Calcular los valores SHAP
print("🧮 Calculando valores SHAP...")
shap_values = explainer.shap_values(X_test_processed)

# Para clasificación binaria, shap_values es una lista con dos arrays
# Usamos el segundo array (clase positiva) para los gráficos
if isinstance(shap_values, list):
    shap_values_class = shap_values[1]  # Clase positiva (pobreza)
else:
    shap_values_class = shap_values

# Crear gráfico de resumen con nombres de características correctos
print("📈 Creando gráfico de resumen SHAP...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class, X_test_processed, feature_names=all_feature_names, plot_type="bar")

# Crear gráfico de resumen detallado
print("📉 Creando gráfico detallado SHAP...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class, X_test_processed, feature_names=all_feature_names)

print("✅ Análisis SHAP completado exitosamente")