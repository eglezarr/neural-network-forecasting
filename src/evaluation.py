"""
evaluation.py — Métricas y guardado estandarizado de resultados
================================================================

UTILIDAD:
    Proporciona funciones comunes para que todos los miembros del equipo
    evalúen sus modelos de la misma forma y guarden los resultados en
    un formato estandarizado. Cada tipo de modelo guarda sus resultados
    en un CSV independiente, evitando conflictos entre compañeros y
    protegiendo los resultados de baselines (que son fijos y definitivos).

OBJETIVO:
    1. Medir MAE (métrica del enunciado) de forma consistente.
    2. Guardar resultados en CSVs separados por tipo de modelo:
       - baseline_results.csv   → fijo, definitivo
       - recurrent_results.csv  → modelos RNN (LSTM/GRU)
       - dense_results.csv      → modelos con capas densas (MLP)
       - cnn_results.csv        → modelos convolucionales
       - mixed_results.csv      → modelos mixtos
    3. Contar parámetros de cada modelo (requisito del enunciado).
    4. Cargar todos los resultados combinados para comparativas finales.

USO TÍPICO:
    from src.evaluation import compute_mae, save_results, count_parameters

    mae_train = compute_mae(y_train, model.predict(X_train))
    mae_test = compute_mae(y_test, model.predict(X_test))

    save_results(
        model_name="LSTM_v1",
        model_type="recurrent",       # Determina en qué CSV se guarda
        input_window=30,
        output_window=5,
        mae_train=mae_train,
        mae_test=mae_test,
        n_params=count_parameters(model)
    )
"""

import numpy as np
import pandas as pd
import os
import glob

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR


# ============================================================================
# MÉTRICAS
# ============================================================================

def compute_mae(y_true, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

    El enunciado especifica MAE como métrica de la competición:
    "Los modelos se entrenarán para minimizar el error absoluto promedio (MAE)."

    Parameters
    ----------
    y_true : np.ndarray
        Valores reales, shape (N, 23).
    y_pred : np.ndarray
        Predicciones del modelo, shape (N, 23).

    Returns
    -------
    float
        MAE promedio sobre todas las muestras y activos.
    """
    return np.mean(np.abs(y_true - y_pred))


# ============================================================================
# CONTEO DE PARÁMETROS
# ============================================================================

def count_parameters(model):
    """
    Cuenta el número total de parámetros entrenables de un modelo Keras.

    El enunciado pide reportar el número de parámetros de cada modelo
    junto con sus resultados.

    Parameters
    ----------
    model : keras.Model
        Modelo de Keras compilado.

    Returns
    -------
    int
        Número total de parámetros entrenables.
    """
    return int(np.sum([
        np.prod(w.shape) for w in model.trainable_weights
    ]))


# ============================================================================
# MAPEO DE TIPO DE MODELO A NOMBRE DE CSV
# ============================================================================

def _get_results_file(model_type):
    """
    Devuelve la ruta del CSV correspondiente al tipo de modelo.

    Parameters
    ----------
    model_type : str
        Tipo de modelo: 'baseline', 'dense', 'recurrent', 'convolutional', 'mixed'.

    Returns
    -------
    str
        Ruta completa al archivo CSV.
    """
    file_map = {
        'baseline': 'baseline_results.csv',
        'dense': 'dense_results.csv',
        'recurrent': 'recurrent_results.csv',
        'convolutional': 'cnn_results.csv',
        'mixed': 'mixed_results.csv',
    }

    filename = file_map.get(model_type, f'{model_type}_results.csv')
    return os.path.join(TABLES_DIR, filename)


# ============================================================================
# COLUMNAS ESTÁNDAR
# ============================================================================

RESULTS_COLUMNS = [
    'model_name',       # Nombre descriptivo (e.g., "LSTM_64units_v1")
    'model_type',       # Tipo: baseline, dense, recurrent, convolutional, mixed
    'input_window',     # Ventana de entrada: 5, 10, 30, 90
    'output_window',    # Ventana de salida: 1, 5, 30, 90
    'mae_train',        # MAE en entrenamiento
    'mae_val',          # MAE en validación (si se usa validation_split)
    'mae_test',         # MAE en test — la métrica de la competición
    'n_params',         # Número de parámetros entrenables
]


# ============================================================================
# GUARDADO DE RESULTADOS
# ============================================================================

def save_results(model_name, model_type, input_window, output_window,
                 mae_train, mae_test, n_params, mae_val=None):
    """
    Guarda los resultados de un modelo en el CSV correspondiente a su tipo.

    El CSV se determina automáticamente por el parámetro model_type:
        - 'baseline'       → baseline_results.csv
        - 'recurrent'      → recurrent_results.csv
        - 'dense'          → dense_results.csv
        - 'convolutional'  → cnn_results.csv
        - 'mixed'          → mixed_results.csv

    Parameters
    ----------
    model_name : str
        Nombre descriptivo del modelo.
    model_type : str
        Tipo de modelo (determina el CSV de destino).
    input_window : int
        Ventana de entrada usada.
    output_window : int
        Ventana de salida usada.
    mae_train : float
        MAE en el conjunto de entrenamiento.
    mae_test : float
        MAE en el conjunto de test.
    n_params : int
        Número de parámetros entrenables del modelo.
    mae_val : float, optional
        MAE en validación.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

    results_file = _get_results_file(model_type)

    new_row = pd.DataFrame([{
        'model_name': model_name,
        'model_type': model_type,
        'input_window': input_window,
        'output_window': output_window,
        'mae_train': round(mae_train, 6),
        'mae_val': round(mae_val, 6) if mae_val is not None else None,
        'mae_test': round(mae_test, 6),
        'n_params': n_params,
    }])

    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(results_file, index=False)
    print(f"Resultados guardados: {model_name} | in={input_window} out={output_window} | MAE test={mae_test:.6f}")


def load_all_results():
    """
    Carga y combina los resultados de TODOS los CSVs de la carpeta tables/.

    Lee todos los archivos *_results.csv y los concatena en un solo
    DataFrame para poder hacer comparativas entre todos los tipos de modelo.

    Returns
    -------
    pd.DataFrame
        DataFrame combinado con todos los resultados, o DataFrame vacío
        si no hay resultados.
    """
    pattern = os.path.join(TABLES_DIR, '*_results.csv')
    files = glob.glob(pattern)

    if not files:
        print("No hay resultados guardados todavía.")
        return pd.DataFrame(columns=RESULTS_COLUMNS)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Cargado: {os.path.basename(f)} ({len(df)} resultados)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined)} resultados combinados")
    return combined


def load_results_by_type(model_type):
    """
    Carga los resultados de un tipo de modelo específico.

    Parameters
    ----------
    model_type : str
        Tipo de modelo: 'baseline', 'dense', 'recurrent', 'convolutional', 'mixed'.

    Returns
    -------
    pd.DataFrame
        DataFrame con los resultados del tipo especificado.
    """
    results_file = _get_results_file(model_type)

    if os.path.exists(results_file):
        return pd.read_csv(results_file)
    else:
        print(f"No hay resultados para el tipo '{model_type}'.")
        return pd.DataFrame(columns=RESULTS_COLUMNS)
