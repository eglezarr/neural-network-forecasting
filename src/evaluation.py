"""
evaluation.py — Métricas y guardado estandarizado de resultados
================================================================

UTILIDAD:
    Proporciona funciones comunes para que todos los miembros del equipo
    evalúen sus modelos de la misma forma y guarden los resultados en
    un formato estandarizado. Esto es crítico porque al final del proyecto
    hay que generar tablas comparativas entre TODOS los modelos (Dense,
    RNN, CNN, Mixtos) — si cada uno guarda resultados en un formato
    distinto, la integración final será un caos.

OBJETIVO:
    1. Medir MAE (métrica del enunciado) de forma consistente.
    2. Guardar resultados de cada modelo en un CSV con columnas fijas.
    3. Contar parámetros de cada modelo (requisito del enunciado).
    4. Cargar todos los resultados para generar tablas y gráficas finales.

USO TÍPICO:
    from src.evaluation import compute_mae, save_results, count_parameters

    mae_train = compute_mae(y_train, model.predict(X_train))
    mae_test = compute_mae(y_test, model.predict(X_test))

    save_results(
        model_name="LSTM_v1",
        model_type="recurrent",
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
# GUARDADO DE RESULTADOS
# ============================================================================

# Columnas del CSV de resultados — formato fijo para todos los modelos
RESULTS_COLUMNS = [
    'model_name',       # Nombre descriptivo (e.g., "LSTM_64units_v1")
    'model_type',       # Tipo: dense, recurrent, convolutional, mixed
    'input_window',     # Ventana de entrada: 5, 10, 30, 90
    'output_window',    # Ventana de salida: 1, 5, 30, 90
    'mae_train',        # MAE en entrenamiento
    'mae_val',          # MAE en validación (si se usa validation_split)
    'mae_test',         # MAE en test — la métrica de la competición
    'n_params',         # Número de parámetros entrenables
]

RESULTS_FILE = os.path.join(TABLES_DIR, 'all_results.csv')


def save_results(model_name, model_type, input_window, output_window,
                 mae_train, mae_test, n_params, mae_val=None):
    """
    Guarda los resultados de un modelo en el CSV centralizado.

    Cada llamada añade una fila al archivo. Si el archivo no existe,
    lo crea con las cabeceras. Esto permite que cada miembro del equipo
    vaya añadiendo sus resultados desde su rama, y al mergear se
    acumulan todos.

    Parameters
    ----------
    model_name : str
        Nombre descriptivo del modelo (e.g., "LSTM_64units_v1").
    model_type : str
        Tipo de modelo: 'dense', 'recurrent', 'convolutional', 'mixed'.
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
        MAE en validación (si se usó validation_split durante el entrenamiento).
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

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

    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(RESULTS_FILE, index=False)
    print(f"Resultados guardados: {model_name} | in={input_window} out={output_window} | MAE test={mae_test:.6f}")


def load_all_results():
    """
    Carga todos los resultados guardados hasta el momento.

    Returns
    -------
    pd.DataFrame
        DataFrame con todos los resultados de todos los modelos,
        o DataFrame vacío si aún no hay resultados.
    """
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    else:
        print("No hay resultados guardados todavía.")
        return pd.DataFrame(columns=RESULTS_COLUMNS)
