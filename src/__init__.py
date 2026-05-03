"""
src/ — Módulo principal del proyecto
=====================================

Este paquete contiene todo el código compartido del equipo.
Cada archivo tiene una responsabilidad clara:

    - data_pipeline.py  → Carga de datos y generación de ventanas temporales
    - evaluation.py     → Métricas (MAE) y guardado estandarizado de resultados
    - plotting.py       → Gráficas: curvas de convergencia, comparativas, matriz
    - baselines.py      → Modelos de referencia: Buy & Hold, Naive
    - portfolio.py      → Construcción de carteras (parte de investigación)

Todos los archivos están diseñados para ser importados desde los notebooks
individuales de cada miembro del equipo:

    from src.data_pipeline import load_data, get_train_test
    from src.evaluation import compute_mae, save_results
    from src.plotting import plot_training_curves
"""
