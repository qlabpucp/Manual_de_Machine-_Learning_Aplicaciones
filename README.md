# Predicción de Ingresos con Regresión Lasso

**Aplicación Práctica para el Manual de Machine Learning en Ciencias Sociales**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josuecaldas/predict_income/blob/main/PYTHON_predict_income.ipynb)

---

### Descripción del Proyecto

Este repositorio contiene la aplicación práctica del modelo de **Regresión Lasso** para predecir el ingreso monetario anual y, fundamentalmente, para identificar los factores socioeconómicos más relevantes que lo determinan. El análisis se basa en un subconjunto de datos de la **Encuesta Nacional de Hogares (ENAHO) 2020** para Lima Metropolitana, Perú.

Este proyecto sirve como un caso de estudio diseñado para ilustrar conceptos clave a estudiantes e investigadores de ciencias sociales que se inician en el uso de técnicas de machine learning.

### Contexto y Justificación Metodológica

En la investigación social, es común enfrentarse a conjuntos de datos con una gran cantidad de variables predictoras (alta dimensionalidad). El desafío consiste en construir modelos que no solo sean precisos, sino también **interpretables**.

La **Regresión Lasso** es particularmente adecuada para esta tarea por dos razones principales:

1.  **Prevención del Sobreajuste:** Su mecanismo de regularización ayuda a crear modelos que generalizan mejor a datos nuevos.
2.  **Selección de Variables:** Su principal ventaja es la capacidad de reducir los coeficientes de las variables menos importantes a exactamente cero. Esto resulta en un modelo más parsimonioso que destaca únicamente los predictores más influyentes, un objetivo central en la investigación social teóricamente informada.

### Resultados Principales

El modelo de Regresión Lasso implementado en el cuaderno de Jupyter (`.ipynb`) arrojó los siguientes resultados clave:

*   **Rendimiento Predictivo:** El modelo final logró un **Coeficiente de Determinación (R²) de 0.50** en el conjunto de datos de prueba. Esto indica que el modelo puede explicar aproximadamente el 50% de la variabilidad en el ingreso, un resultado sólido para datos socioeconómicos complejos.

*   **Selección de Variables Relevantes:** De las 144 variables iniciales, el modelo seleccionó un subconjunto mucho más pequeño como predictores significativos. Entre los factores con mayor impacto positivo en el ingreso se encuentran:
    *   Años de Estudio Aprobados
    *   Edad
    *   Haber trabajado la última semana
    *   Estar afiliado a un sistema privado de pensiones (AFP)

### Estructura del Repositorio

*   `PYTHON_predict_income.ipynb`: El cuaderno de Jupyter/Colab con el código completo del análisis, desde la carga de datos hasta la visualización de resultados.
*   `predict_income_2020.csv`: El conjunto de datos utilizado en el análisis.
*   `README.md`: Este archivo, que proporciona una visión general del proyecto.

### ¿Cómo Usar este Repositorio?

#### Opción 1 (Recomendada): Abrir en Google Colab

La forma más sencilla de explorar y ejecutar el análisis es a través de Google Colab, que no requiere ninguna instalación en tu computadora.

1.  Simplemente haz clic en el siguiente botón:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/josuecaldas/predict_income/blob/main/PYTHON_predict_income.ipynb)
2.  El cuaderno se abrirá en tu navegador y podrás ejecutar cada celda de código de forma interactiva.

#### Opción 2: Ejecución en un Entorno Local

Si prefieres trabajar en tu propia máquina:

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/josuecaldas/predict_income.git
    ```
2.  **Navega al directorio del proyecto:**
    ```bash
    cd predict_income
    ```
3.  **Instala las dependencias necesarias:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib jupyter
    ```
4.  **Inicia Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Desde la interfaz de Jupyter en tu navegador, abre el archivo `PYTHON_predict_income.ipynb`.

### Fuente de los Datos

Los datos son un subconjunto de la Encuesta Nacional de Hogares (ENAHO) 2020, realizada por el Instituto Nacional de Estadística e Informática (INEI) del Perú.

### Autor

Josué Caldas