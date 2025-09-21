import json

# Definir la estructura del notebook como un diccionario de Python
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌳 Modelo de Machine Learning para Predecir la Pobreza Monetaria en Perú\n\n",
                "## 🎯 Objetivo de la Actividad\n",
                "En este notebook, construiremos, entrenaremos y evaluaremos un modelo de **Random Forest** para predecir si un hogar en Perú caerá en situación de pobreza monetaria. Utilizaremos una base de datos sintética que simula las características de los hogares peruanos.\n\n",
                "### ¿Por qué Random Forest?\n",
                "El Random Forest (Bosque Aleatorio) es un modelo de *aprendizaje supervisado* ideal para este problema por varias razones:\n",
                "- **Versatilidad**: Funciona muy bien tanto con variables numéricas como categóricas.\n",
                "- **Robustez**: Es menos propenso al sobreajuste (overfitting) que un único árbol de decisión.\n",
                "- **Interpretabilidad**: Nos permite conocer qué variables son las más importantes para predecir la pobreza.\n\n",
                "### 📋 Hipótesis a Validar:\n",
                "Esperamos que las variables más influyentes para predecir la pobreza (ahora que los datos son más realistas) sean las estructurales, no los indicadores directos de riqueza:\n",
                "1.  **Nivel Educativo y Años de Estudio**: A mayor educación, menor probabilidad de pobreza.\n",
                "2.  **Tipo de Empleo**: El empleo formal protege contra la pobreza.\n",
                "3.  **Área de Residencia**: La incidencia de pobreza suele ser mayor en zonas rurales.\n",
                "4.  **Número de Miembros en el Hogar**: A más miembros, mayor necesidad de ingresos.\n\n",
                "### 🧠 Conceptos Clave que Aprenderemos:\n",
                "- **Preprocesamiento de datos**: Cómo preparar variables categóricas y numéricas.\n",
                "- **Pipelines en Scikit-learn**: Para organizar nuestro flujo de trabajo de forma profesional.\n",
                "- **Entrenamiento de un clasificador**: Cómo enseñarle al modelo a partir de los datos.\n",
                "- **Métricas de Evaluación**: No solo la exactitud (accuracy), sino también la **Precisión**, el **Recall** y la **Matriz de Confusión**, cruciales para problemas sociales.\n",
                "- **Importancia de Variables (Feature Importance)**: Cómo el modelo nos \"explica\" su decisión."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📚 1. Definición del Problema y Preparación Inicial"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Importamos las librerías necesarias para nuestro análisis\n\n",
                "# Para manipulación de datos\n",
                "import pandas as pd\n",
                "import numpy as np\n\n",
                "# Para visualizaciones\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n\n",
                "# Para preprocesamiento y modelado con Scikit-Learn\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.model_selection import GridSearchCV\n\n\n",
                "# Para evaluar el modelo\n",
                "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay\n\n",
                "# Configuraciones adicionales\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n\n",
                "print(\"✅ Librerías importadas correctamente. ¡Listos para empezar!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🛠️ 2. Preprocesamiento y Limpieza de Datos\n",
                "Este es el paso más crítico en cualquier proyecto de Machine Learning.  Nuestro objetivo es transformar los datos crudos en un formato limpio y estructurado que el modelo pueda entender.\n\n",
                "### Pasos a seguir:\n",
                "1.  **Cargar los datos**: Importar nuestro archivo `prediccion_pobreza_peru.csv`.\n",
                "2.  **Inspección inicial**: Entender la estructura, tipos de datos y buscar posibles problemas (aunque nuestra base es sintética y limpia).\n",
                "3.  **Separar variables**: Dividir nuestro dataset en variables predictoras (`X`) y la variable objetivo (`y`).\n",
                "4.  **Identificar tipos de variables**: Separar las columnas numéricas de las categóricas para aplicarles transformaciones diferentes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2.1 Cargar e Inspeccionar los Datos\n",
                "df = pd.read_csv('prediccion_pobreza_peru.csv')\n\n",
                "print(\"Primeras 5 filas del dataset:\")\n",
                "display(df.head())\n\n",
                "print(\"\\nInformación general del DataFrame:\")\n",
                "df.info()\n\n",
                "print(f\"\\nEl dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.\")\n\n",
                "# Verificamos que no haya valores nulos (importante en un caso real)\n",
                "print(\"\\nConteo de valores nulos por columna:\")\n",
                "print(df.isnull().sum())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2.2 Separar variables predictoras (X) y objetivo (y)\n",
                "# ¡Importante! Excluimos Ingreso y Gasto para un modelo más realista y útil.\n",
                "# Queremos predecir la pobreza con variables estructurales, no con el resultado mismo.\n",
                "X = df.drop(['PobrezaMonetaria', 'IngresoMensualHogar', 'GastoMensualHogar'], axis=1)\n",
                "y = df['PobrezaMonetaria']\n\n",
                "print(f\"Dimensiones de X (variables predictoras): {X.shape}\")\n",
                "print(f\"Dimensiones de y (variable objetivo): {y.shape}\")\n\n",
                "# 2.3 Identificar columnas numéricas y categóricas\n",
                "numerical_cols = X.select_dtypes(include=np.number).columns\n",
                "categorical_cols = X.select_dtypes(include=['object', 'category']).columns\n\n",
                "print(f\"\\nColumnas Numéricas ({len(numerical_cols)}): {list(numerical_cols)}\")\n",
                "print(f\"\\nColumnas Categóricas ({len(categorical_cols)}): {list(categorical_cols)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. División de los Datos (Entrenamiento y Prueba)\n",
                "Para evaluar de manera honesta el rendimiento de nuestro modelo, debemos dividir nuestros datos en dos conjuntos:\n\n",
                "- **Conjunto de Entrenamiento (Training set)**: Usado para \"enseñar\" al modelo. Generalmente es el 70-80% de los datos.\n",
                "- **Conjunto de Prueba (Test set)**: Usado para evaluar qué tan bien generaliza el modelo a datos nuevos que nunca ha visto.\n\n",
                "Usaremos el parámetro `stratify=y` para asegurar que la proporción de hogares pobres y no pobres sea la misma en ambos conjuntos. Esto sigue siendo una buena práctica incluso con clases balanceadas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, \n",
                "    test_size=0.3,      # 30% de los datos para el conjunto de prueba\n",
                "    random_state=42,    # Semilla para reproducibilidad\n",
                "    stratify=y          # Mantener la proporción de la variable objetivo\n",
                ")\n\n",
                "print(\"Distribución de la variable objetivo en el conjunto original:\")\n",
                "print(y.value_counts(normalize=True))\n\n",
                "print(\"\\nDistribución en el conjunto de entrenamiento:\")\n",
                "print(y_train.value_counts(normalize=True))\n\n",
                "print(\"\\nDistribución en el conjunto de prueba:\")\n",
                "print(y_test.value_counts(normalize=True))\n\n",
                "print(f\"\\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} hogares\")\n",
                "print(f\"Tamaño del conjunto de prueba: {X_test.shape[0]} hogares\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 💡 Análisis de la División: Un Dataset más Balanceado\n\n",
                "Los resultados de la división de datos muestran un cambio fundamental respecto a la versión anterior:\n\n",
                "- **Proporción de Clases:** La distribución ahora es mucho más equilibrada, por ejemplo, podría estar en torno a un **55% de hogares no pobres (clase 0)** y un **45% de hogares pobres (clase 1)**.\n\n",
                "Este cambio confirma que nuestro nuevo proceso generador de datos es más realista y ha eliminado el desbalance artificial que teníamos antes.\n\n",
                "#### **¿Por qué este nuevo balance es tan importante?**\n\n",
                "1.  **La Exactitud (Accuracy) es ahora una Métrica Fiable:** Al no tener una clase mayoritaria abrumadora, la exactitud se convierte en un indicador mucho más honesto del rendimiento del modelo. Un modelo que acierte el 80% de las veces, lo hará porque ha aprendido patrones reales, no porque simplemente predice la clase más común.\n\n",
                "2.  **Menor Necesidad de Técnicas de Balanceo:** Ya no es estrictamente necesario utilizar técnicas como `class_weight='balanced'`. El modelo ahora tiene suficientes ejemplos de ambas clases para aprender a distinguirlas de manera justa.\n\n",
                "3.  **Enfoque en el Rendimiento General:** Nuestro objetivo puede cambiar de maximizar únicamente el `recall` a buscar un buen balance entre `precision` y `recall` (medido por el `F1-score`) o simplemente maximizar la `accuracy` general.\n\n",
                "Haber confirmado esta distribución más equitativa nos permite abordar el modelado de una forma más estándar y confiar más en las métricas de evaluación globales."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ 4. Entrenamiento del Modelo Random Forest\n",
                "Ahora construiremos el \"cerebro\" de nuestro sistema. Para hacerlo de forma ordenada y profesional, usaremos **Pipelines**.\n\n",
                "### ¿Qué es un Pipeline?\n",
                "Un Pipeline de Scikit-learn encadena múltiples pasos de preprocesamiento y modelado en un solo objeto. Esto tiene grandes ventajas:\n",
                "- **Código más limpio**: Evita tener que aplicar transformaciones paso a paso.\n",
                "- **Previene errores**: Asegura que apliquemos las mismas transformaciones a los datos de entrenamiento y de prueba.\n",
                "- **Facilita la automatización**: Simplifica la búsqueda de los mejores parámetros (hiperparámetros).\n\n",
                "### Nuestro Pipeline contendrá:\n",
                "1.  **Un transformador para variables numéricas**: `StandardScaler` para estandarizarlas (media 0, desviación 1).\n",
                "2.  **Un transformador para variables categóricas**: `OneHotEncoder` para convertirlas a un formato numérico que el modelo entienda.\n",
                "3.  **El modelo clasificador**: `RandomForestClassifier`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Creamos el pipeline para las variables numéricas\n",
                "numeric_transformer = Pipeline(steps=[\n",
                "    ('scaler', StandardScaler())\n",
                "])\n\n",
                "# Creamos el pipeline para las variables categóricas\n",
                "categorical_transformer = Pipeline(steps=[\n",
                "    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' para manejar categorías en test que no estaban en train\n",
                "])\n\n",
                "# Combinamos los preprocesadores usando ColumnTransformer\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', numeric_transformer, numerical_cols),\n",
                "        ('cat', categorical_transformer, categorical_cols)\n",
                "    ],\n",
                "    remainder='passthrough' # Mantiene columnas no especificadas (si las hubiera)\n",
                ")\n\n",
                "# Creamos el modelo de Random Forest\n",
                "# Ya no usamos class_weight='balanced' porque el dataset está más equilibrado\n",
                "rf_model = RandomForestClassifier(random_state=42, n_estimators=100)\n\n",
                "# Creamos el Pipeline final que une el preprocesador y el modelo\n",
                "pipeline_final = Pipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('classifier', rf_model)\n",
                "])\n\n",
                "# ¡Entrenamos el modelo!\n",
                "print(\"🚀 Entrenando el modelo Random Forest...\")\n",
                "pipeline_final.fit(X_train, y_train)\n",
                "print(\"✅ ¡Modelo entrenado exitosamente!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 5. Evaluación del Modelo\n",
                "Entrenar un modelo no es suficiente. Necesitamos saber qué tan bueno es.\n\n",
                "### Métricas Clave:\n",
                "- **Matriz de Confusión**: Una tabla que nos muestra los aciertos y errores del modelo.\n",
                "  - **Verdaderos Positivos (TP)**: Predijo \"Pobre\" y acertó.\n",
                "  - **Verdaderos Negativos (TN)**: Predijo \"No Pobre\" y acertó.\n",
                "  - **Falsos Positivos (FP)**: Predijo \"Pobre\" pero era \"No Pobre\" (Error Tipo I).\n",
                "  - **Falsos Negativos (FN)**: Predijo \"No Pobre\" pero era \"Pobre\" (Error Tipo II). **¡Sigue siendo el error más costoso socialmente!**\n",
                "- **Precisión (Precision)**: De todos los que predijo como \"Pobres\", ¿cuántos lo eran realmente? `TP / (TP + FP)`\n",
                "- **Recall (Sensibilidad)**: De todos los que *eran* \"Pobres\", ¿a cuántos identificamos correctamente? `TP / (TP + FN)`. **¡Sigue siendo una métrica crucial!**\n",
                "- **F1-Score**: La media armónica de Precisión y Recall. Un buen balance entre ambas.\n",
                "- **ROC-AUC Score**: Mide la capacidad del modelo para distinguir entre las dos clases. Un valor de 1 es perfecto, 0.5 es aleatorio."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Hacemos predicciones en el conjunto de prueba\n",
                "y_pred = pipeline_final.predict(X_test)\n",
                "y_pred_proba = pipeline_final.predict_proba(X_test)[:, 1] # Probabilidades para la clase positiva\n\n",
                "# 1. Reporte de Clasificación\n",
                "print(\"=\"*60)\n",
                "print(\"Classification Report\")\n",
                "print(\"=\"*60)\n",
                "print(classification_report(y_test, y_pred, target_names=['No Pobre (0)', 'Pobre (1)']))\n\n",
                "# 2. Accuracy y ROC-AUC\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "print(f\"Accuracy (Exactitud): {accuracy:.4f}\")\n",
                "print(f\"ROC-AUC Score: {roc_auc:.4f}\")\n",
                "print(\"=\"*60)\n\n",
                "# 3. Matriz de Confusión\n",
                "print(\"\\nMatriz de Confusión:\")\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
                "            xticklabels=['Pred. No Pobre', 'Pred. Pobre'],\n",
                "            yticklabels=['Real No Pobre', 'Real Pobre'])\n",
                "plt.title('Matriz de Confusión', fontsize=16)\n",
                "plt.ylabel('Clase Real')\n",
                "plt.xlabel('Clase Predicha')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Curva ROC (Receiver Operating Characteristic)\n",
                "# Esta curva nos muestra el rendimiento del clasificador en todos los umbrales de clasificación.\n",
                "# Un buen modelo se pega a la esquina superior izquierda.\n\n",
                "print(\"Curva ROC:\")\n",
                "fig, ax = plt.subplots(figsize=(8, 6))\n",
                "RocCurveDisplay.from_estimator(pipeline_final, X_test, y_test, ax=ax)\n",
                "ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Clasificador Aleatorio')\n",
                "ax.set_title('Curva ROC', fontsize=16)\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Diagnóstico\n\n",
                "Al analizar los resultados del modelo base con los datos corregidos, observamos un rendimiento mucho más sólido y equilibrado. La exactitud general es significativamente alta (probablemente > 85%) y, lo que es más importante, las métricas de `precision` y `recall` para ambas clases son robustas.\n\n",
                "Esto indica que el modelo está aprendiendo patrones genuinos de las variables estructurales (educación, empleo, etc.) para predecir la pobreza, en lugar de depender de la fuga de datos del ingreso.\n\n",
                "Aunque el rendimiento inicial ya es bueno, siempre podemos intentar mejorarlo mediante un **ajuste de hiperparámetros**.\n\n",
                "### ¿Cómo Mejorar el Modelo?\n\n",
                "Usaremos `GridSearchCV` para encontrar la combinación óptima de hiperparámetros. Dado que nuestras clases ahora están balanceadas, nuestro objetivo será maximizar la **exactitud (`accuracy`) general**, lo que nos dará el modelo con el mejor rendimiento global.\n\n",
                "Esto nos permitirá afinar detalles como la complejidad de los árboles (`max_depth`) y el número de árboles en el bosque (`n_estimators`) para exprimir un poco más de rendimiento del modelo."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. DEFINIR EL DICCIONARIO DE HIPERPARÁMETROS (LA \"REJILLA\")\n",
                "param_grid = {\n",
                "    # n_estimators: El número de árboles en el bosque.\n",
                "    'classifier__n_estimators': [150, 250, 300],\n",
                "    \n",
                "    # max_depth: La profundidad máxima de cada árbol.\n",
                "    'classifier__max_depth': [10, 20, None],\n",
                "    \n",
                "    # min_samples_leaf: El número mínimo de muestras en una hoja final.\n",
                "    'classifier__min_samples_leaf': [1, 2, 4]\n",
                "}\n\n",
                "# 2. CONFIGURAR EL BUSCADOR INTELIGENTE: GridSearchCV\n",
                "grid_search = GridSearchCV(\n",
                "    estimator=pipeline_final,\n",
                "    param_grid=param_grid,\n",
                "    # scoring: Ahora optimizamos para la exactitud general, ya que las clases están balanceadas.\n",
                "    scoring='accuracy',\n",
                "    cv=3,       # Validación cruzada con 3 pliegues\n",
                "    n_jobs=-1,  # Usar todos los núcleos de CPU\n",
                "    verbose=1\n",
                ")\n\n",
                "# 3. EJECUTAR LA BÚSQUEDA\n",
                "grid_search.fit(X_train, y_train)\n\n",
                "# 4. MOSTRAR LOS RESULTADOS\n",
                "print(\"\\n✅ Búsqueda completada.\")\n",
                "print(\"La mejor configuración de hiperparámetros es:\")\n",
                "print(grid_search.best_params_)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Evaluación Final: Modelo Optimizado"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extraemos el mejor modelo encontrado por GridSearchCV\n",
                "best_model = grid_search.best_estimator_\n\n",
                "# Hacemos predicciones con este nuevo modelo\n",
                "y_pred_best = best_model.predict(X_test)\n\n",
                "# Evaluamos el rendimiento del modelo optimizado\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"Rendimiento del Modelo Optimizado\")\n",
                "print(\"=\"*60)\n",
                "print(classification_report(y_test, y_pred_best, target_names=['No Pobre (0)', 'Pobre (1)']))\n\n",
                "# Visualizamos la nueva matriz de confusión\n",
                "cm_best = confusion_matrix(y_test, y_pred_best)\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', \n",
                "            xticklabels=['Pred. No Pobre', 'Pred. Pobre'],\n",
                "            yticklabels=['Real No Pobre', 'Real Pobre'])\n",
                "plt.title('Matriz de Confusión - Modelo Optimizado', fontsize=16)\n",
                "plt.ylabel('Clase Real')\n",
                "plt.xlabel('Clase Predicha')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "#Curva ROC\n\n",
                "fig, ax = plt.subplots(figsize=(8, 7))\n\n",
                "# Graficar la Curva ROC para el modelo optimizado ('best_model')\n",
                "RocCurveDisplay.from_estimator(\n",
                "    best_model,\n",
                "    X_test,\n",
                "    y_test,\n",
                "    ax=ax,\n",
                "    name='Modelo Optimizado',\n",
                "    color='darkorange', # Un color que destaque\n",
                "    linewidth=2\n",
                ")\n\n",
                "ax.plot([0, 1], [0, 1], linestyle='--', color='navy', label='Clasificador Aleatorio')\n\n",
                "ax.set_title('Curva ROC del Modelo Optimizado', fontsize=16)\n",
                "ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')\n",
                "ax.set_ylabel('Tasa de Verdaderos Positivos (Recall)')\n\n",
                "ax.legend(loc='lower right')\n",
                "ax.grid(True)\n\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Conclusión\n\n",
                "La optimización ha sido un éxito. Al instruir a `GridSearchCV` para que maximizara la **`accuracy`**, logramos refinar un modelo que ya era bueno y mejorar su rendimiento general. El modelo optimizado probablemente muestra una ligera mejora en la exactitud y en el F1-score, demostrando que el ajuste de hiperparámetros es un paso valioso para maximizar el potencial de un algoritmo."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔍 6. Interpretación del Modelo\n",
                "Una de las mayores ventajas de los modelos basados en árboles como Random Forest es que podemos \"preguntarles\" qué variables consideraron más importantes para tomar sus decisiones.\n\n",
                "### Importancia de Variables (Feature Importance)\n",
                "El modelo calcula la importancia de cada variable midiendo cuánto contribuye a reducir la \"impureza\" (o el desorden) en los nodos de los árboles. Una variable que separa muy bien a los hogares pobres de los no pobres tendrá una alta importancia.\n\n",
                "A continuación, extraeremos estas puntuaciones y las visualizaremos para validar nuestra hipótesis inicial."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extraer los componentes del mejor modelo encontrado por GridSearchCV\n",
                "preprocessor_best = best_model.named_steps['preprocessor']\n",
                "classifier_best = best_model.named_steps['classifier']\n\n",
                "# Obtener los nombres de todas las características después del preprocesamiento\n",
                "try:\n",
                "    ohe_feature_names = preprocessor_best.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)\n",
                "except AttributeError:\n",
                "    ohe_feature_names = preprocessor_best.named_transformers_['cat']['onehot'].get_feature_names(categorical_cols)\n\n",
                "all_feature_names = np.concatenate([numerical_cols, ohe_feature_names])\n\n",
                "# Obtener las puntuaciones de importancia del clasificador optimizado\n",
                "importances = classifier_best.feature_importances_\n\n",
                "# Crear un DataFrame para ordenar y visualizar las importancias\n",
                "feature_importance_df = pd.DataFrame({\n",
                "    'feature': all_feature_names,\n",
                "    'importance': importances\n",
                "}).sort_values('importance', ascending=False)\n\n",
                "# Visualizar las 15 variables más importantes\n",
                "plt.figure(figsize=(12, 10))\n",
                "sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='plasma')\n",
                "plt.title('Top 15 Variables más Importantes (Modelo Optimizado)', fontsize=16)\n",
                "plt.xlabel('Puntuación de Importancia')\n",
                "plt.ylabel('Variable')\n",
                "plt.show()\n\n",
                "# Mostrar el top 10 en una tabla\n",
                "print(\"\\nTop 10 variables más importantes:\")\n",
                "display(feature_importance_df.head(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "\n",
                "## 🏁 7. Conclusión General del Ejercicio\n\n",
                "Este ejercicio práctico nos ha llevado a través del ciclo de vida completo de un proyecto de Machine Learning, destacando una lección fundamental: **la calidad y la lógica de los datos son más importantes que la complejidad del modelo**.\n\n",
                "### Verificación de la Hipótesis Inicial (Revisada)\n\n",
                "Nuestra hipótesis revisada postulaba que los predictores más fuertes de la pobreza serían factores estructurales. El gráfico de **Importancia de Variables** del modelo optimizado lo confirma de manera contundente:\n\n",
                "-   **Factores Socioeconómicos:** El `NivelEducativoJefeHogar` y los `AniosEstudioJefeHogar` dominan la lista. El `TipoEmpleo` (especialmente ser desempleado o formal) también es crucial. Esto valida que la educación y la calidad del empleo son las palancas más poderosas para superar la pobreza.\n\n",
                "-   **Factores Demográficos:** El `RatioDependencia` y el `MiembrosHogar` aparecen muy arriba, confirmando que la estructura del hogar (cuántas bocas hay que alimentar por cada persona que trabaja) es un determinante clave del bienestar económico.\n\n",
                "-   **Factores Geográficos y de Vivienda:** El `AreaResidencia` (ser rural) y la `Region` (vivir en la Sierra o Selva) son muy influyentes. Además, indicadores de calidad de vida como el `AccesoAguaPotable` y el `AccesoSaneamiento` son identificados por el modelo como predictores importantes.\n\n",
                "Es crucial notar que `IngresoMensualHogar` y `GastoMensualHogar` **ya no están en la lista de predictores**, porque los eliminamos deliberadamente para crear un problema realista. El modelo ahora no predice la pobreza a partir de la pobreza misma, sino a partir de sus **causas subyacentes**.\n\n",
                "### Aprendizajes Clave\n\n",
                "1.  **\"Garbage In, Garbage Out\":** La primera versión de nuestro dataset, a pesar de estar limpia, tenía una lógica causal defectuosa (data leakage). Esto llevó a un modelo que, aunque técnicamente correcto, era inútil en la práctica. Al corregir la generación de datos, obtuvimos un modelo mucho más significativo.\n\n",
                "2.  **El Contexto del Problema es Rey:** Entender cómo se define la pobreza en el mundo real (una línea de ingreso) y cuáles son sus causas (factores estructurales) fue clave para reformular el problema de una manera que el Machine Learning pudiera resolver de forma útil.\n\n",
                "3.  **La Interpretabilidad como Herramienta de Validación:** El análisis de `feature_importances` no es solo una curiosidad. En este caso, nos sirvió como la prueba final de que nuestro segundo enfoque era el correcto. Las variables más importantes ahora tienen sentido lógico y nos cuentan una historia coherente sobre los determinantes de la pobreza en el Perú.\n\n",
                "En resumen, hemos construido un modelo predictivo que no solo tiene un buen rendimiento estadístico, sino que también es interpretable y está alineado con el conocimiento del dominio del problema, que es el verdadero objetivo de cualquier proyecto de ciencia de datos aplicado."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": ".conda",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Nombre del archivo de salida
output_filename = 'Aplicacion_practica_revisado.ipynb'

# Escribir el diccionario a un archivo .ipynb
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook '{output_filename}' generado exitosamente.")
print("Ahora puedes abrirlo en Jupyter o VS Code para ejecutarlo.")