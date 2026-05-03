"""
config.py — Constantes compartidas del proyecto
=================================================

UTILIDAD:
    Este archivo centraliza TODAS las constantes que usa el proyecto.
    De esta forma, si hay que cambiar una ventana temporal, un seed o un ticker,
    se cambia aquí y se propaga automáticamente a todo el código.
    Evita que cada compañero hardcodee valores distintos en su notebook.

OBJETIVO:
    Garantizar que los 3 miembros del equipo trabajen con exactamente
    los mismos parámetros: mismas ventanas, mismo split, misma semilla,
    mismos tickers. Si esto no está centralizado, los resultados no
    serán comparables.
"""

# ============================================================================
# TICKERS — Los 23 activos del S&P 500 definidos por el profesor
# ============================================================================
TICKERS = [
    'AEP', 'BA', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'GD', 'GE',
    'HON', 'HPQ', 'IBM', 'IP', 'JNJ', 'KO', 'KR', 'MMM', 'MO', 'MRK',
    'MSI', 'PG', 'XOM'
]

# ============================================================================
# VENTANAS TEMPORALES — Definidas en el enunciado (sección 4)
# ============================================================================
INPUT_WINDOWS = [5, 10, 30, 90]     # Ventanas de entrada (días de historia)
OUTPUT_WINDOWS = [1, 5, 30, 90]     # Ventanas de salida (días a futuro)

# ============================================================================
# PARÁMETROS DE SPLIT — Del notebook del profesor
# ============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.1         # 90% train, 10% test
START_DATE = '1945-01-01'

# ============================================================================
# TIPOS DE MODELO — Para estandarizar nombres en resultados y gráficas
# ============================================================================
MODEL_TYPES = ['dense', 'recurrent', 'convolutional', 'mixed']

# ============================================================================
# RUTAS — Para que todos guarden resultados en el mismo sitio
# ============================================================================
RESULTS_DIR = 'results/'
TABLES_DIR = 'results/tables/'
FIGURES_DIR = 'results/figures/'
MODELS_DIR = 'models/'
