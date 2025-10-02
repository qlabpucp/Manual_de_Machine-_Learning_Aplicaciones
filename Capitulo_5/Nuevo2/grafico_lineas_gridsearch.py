# Gráfico de líneas para visualizar resultados de GridSearchCV

# Extraemos los resultados de GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

# Creamos un DataFrame con las columnas necesarias usando los nombres originales
# Mapeamos los parámetros de Random Forest a los nombres originales solicitados
plot_data = pd.DataFrame({
    'nodesize': results['param_classifier__min_samples_leaf'],
    'R2': results['mean_test_score'],
    'pmtry': results['param_classifier__max_depth'],
    'ntree': results['param_classifier__n_estimators']
})

# Convertimos los valores None a una representación numérica para el gráfico
plot_data['pmtry'] = plot_data['pmtry'].fillna(30)  # Usamos 30 para representar None (sin límite)

# Creamos el gráfico de líneas
plt.figure(figsize=(12, 8))

# Usamos seaborn lineplot con los parámetros especificados
ax = sns.lineplot(
    data=plot_data,
    x='nodesize',
    y='R2',
    hue='pmtry',
    style='ntree',
    markers=True,
    dashes=False,
    palette='viridis',
    markersize=10,
    linewidth=2.5
)

# Configuramos las etiquetas de los ejes
plt.xlabel('nodesize', fontsize=14)
plt.ylabel('R2 (Accuracy)', fontsize=14)

# Configuramos el título
plt.title('Resultados de GridSearchCV: Rendimiento vs Hiperparámetros', fontsize=16)

# Mejoramos la leyenda
handles, labels = ax.get_legend_handles_labels()
# Modificamos las etiquetas para mostrar None en lugar de 30
labels = ['None' if label == '30' else label for label in labels]
ax.legend(handles, labels, title='pmtry / ntree', bbox_to_anchor=(1.05, 1), loc='upper left')

# Añadimos una cuadrícula para mejor legibilidad
plt.grid(True, linestyle='--', alpha=0.7)

# Ajustamos el layout para evitar que la leyenda se corte
plt.tight_layout()

# Guardamos el gráfico
plt.savefig('gridsearch_lineas_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# Mostramos una tabla con los datos utilizados para el gráfico
print("Datos utilizados para el gráfico:")
display(plot_data.sort_values(['pmtry', 'ntree', 'nodesize']))