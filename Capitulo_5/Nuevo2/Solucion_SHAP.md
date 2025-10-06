# Solución para el Problema con SHAP

## Problema Identificado

El error en las celdas finales del notebook `Aplicacion_practica_revisado.ipynb` está relacionado con la implementación de SHAP. El error específico es:

```
NameError: name 'best_model' is not defined
```

## Causas del Problema

1. **Variable no definida**: La variable `best_model` no está definida en el contexto actual cuando se ejecutan las celdas SHAP.

2. **Datos sin preprocesar**: SHAP está recibiendo `X_test` en formato original, pero el modelo espera datos preprocesados.

3. **Explicador incorrecto**: `TreeExplainer` está recibiendo el pipeline completo en lugar del clasificador.

4. **Nombres de características incorrectos**: SHAP no tiene los nombres correctos de las características después del preprocesamiento.

## Solución Propuesta

### Opción 1: Usar el notebook SHAP_corregido.ipynb

Hemos creado un notebook `SHAP_corregido.ipynb` que contiene el código corregido y puede ser ejecutado de forma independiente.

### Opción 2: Reemplazar las celdas SHAP en el notebook original

Reemplaza las celdas SHAP (líneas 764-776) en el notebook original con el siguiente código:

```python
# Instalar SHAP si no está instalado
try:
    import shap
except ImportError:
    !pip install shap
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
```

## Cambios Clave en la Solución

1. **Verificación de variables**: Se verifica si las variables necesarias están definidas antes de usarlas.

2. **Datos preprocesados**: Se preprocesan los datos de prueba antes de pasarlos a SHAP.

3. **Explicador correcto**: Se usa solo el clasificador (`best_model.named_steps['classifier']`) en lugar del pipeline completo.

4. **Nombres de características**: Se proporcionan los nombres correctos de las características a los gráficos de SHAP.

5. **Manejo de clasificación binaria**: Se maneja correctamente el caso de clasificación binaria donde `shap_values` es una lista.

## Recomendación

La opción más sencilla es usar el notebook `SHAP_corregido.ipynb` que hemos creado, ya que contiene todo el código corregido y puede ser ejecutado de forma independiente.

Si prefieres mantener todo en un solo notebook, copia y pega el código proporcionado en la sección "Opción 2" en las celdas finales del notebook original.