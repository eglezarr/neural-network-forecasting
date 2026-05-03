"""
baselines.py — Modelos de referencia (Buy & Hold + Naive)
==========================================================

UTILIDAD:
    Proporciona modelos simples que sirven como SUELO MÍNIMO de rendimiento.
    Cualquier red neuronal que no bata a estos modelos no está aportando
    valor. El enunciado lo pide explícitamente:
    "Se deben añadir modelos simples, por ejemplo Buy and Hold."

OBJETIVO:
    1. Buy & Hold: simula mantener la posición sin hacer nada.
       En el contexto de log-returns, esto equivale a usar la media
       histórica de los returns como predicción.
    2. Naive (Last Value): predice que el futuro será igual al último
       valor observado. Es el benchmark más básico posible.
    3. Zero Prediction: predice que el log-return futuro promedio será
       cero (los log-returns diarios tienen media cercana a cero).

    Estos baselines se evalúan con MAE para cada combinación de ventanas,
    y sus resultados se guardan en el mismo CSV que los modelos de redes
    neuronales para poder compararlos directamente.

USO TÍPICO:
    from src.baselines import evaluate_all_baselines

    returns = load_data()
    evaluate_all_baselines(returns)  # evalúa y guarda todo automáticamente
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_WINDOWS, OUTPUT_WINDOWS
from src.data_pipeline import get_train_test
from src.evaluation import compute_mae, save_results


# ============================================================================
# BASELINE 1: NAIVE (Last Value)
# ============================================================================

def predict_naive(X):
    """
    Predice que el promedio futuro de los log-returns será igual al
    último valor de la ventana de entrada.

    Lógica: si los últimos returns son los del día más reciente,
    el modelo naive asume que mañana será como hoy.

    Parameters
    ----------
    X : np.ndarray
        Secuencias de entrada, shape (N, input_window, 23).

    Returns
    -------
    np.ndarray
        Predicciones, shape (N, 23). Último timestep de cada secuencia.
    """
    return X[:, -1, :]  # Último día de la ventana de entrada


# ============================================================================
# BASELINE 2: ZERO PREDICTION
# ============================================================================

def predict_zero(X):
    """
    Predice que el promedio futuro de los log-returns será cero.

    Justificación: los log-returns diarios tienen media muy cercana a cero.
    Este es el benchmark más ingenuo pero sorprendentemente difícil de batir
    en horizontes cortos.

    Parameters
    ----------
    X : np.ndarray
        Secuencias de entrada, shape (N, input_window, 23).

    Returns
    -------
    np.ndarray
        Array de ceros, shape (N, 23).
    """
    return np.zeros((X.shape[0], X.shape[2]))


# ============================================================================
# BASELINE 3: MEDIA DE LA VENTANA DE ENTRADA
# ============================================================================

def predict_mean(X):
    """
    Predice que el promedio futuro será igual a la media de la ventana
    de entrada. Es una versión suavizada del naive.

    Parameters
    ----------
    X : np.ndarray
        Secuencias de entrada, shape (N, input_window, 23).

    Returns
    -------
    np.ndarray
        Media sobre la dimensión temporal, shape (N, 23).
    """
    return np.mean(X, axis=1)


# ============================================================================
# EVALUACIÓN AUTOMÁTICA DE TODOS LOS BASELINES
# ============================================================================

def evaluate_all_baselines(returns):
    """
    Evalúa los 3 modelos baseline para las 16 combinaciones de ventanas
    y guarda los resultados en el CSV centralizado.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-returns diarios (output de load_data()).
    """
    baselines = {
        'Naive_LastValue': predict_naive,
        'Zero_Prediction': predict_zero,
        'Mean_Window': predict_mean,
    }

    print("=" * 60)
    print("Evaluando baselines para todas las combinaciones de ventanas...")
    print("=" * 60)

    for iw in INPUT_WINDOWS:
        for ow in OUTPUT_WINDOWS:
            X_train, X_test, y_train, y_test = get_train_test(returns, iw, ow)

            for name, predict_fn in baselines.items():
                y_pred_train = predict_fn(X_train)
                y_pred_test = predict_fn(X_test)

                mae_train = compute_mae(y_train, y_pred_train)
                mae_test = compute_mae(y_test, y_pred_test)

                save_results(
                    model_name=name,
                    model_type='baseline',
                    input_window=iw,
                    output_window=ow,
                    mae_train=mae_train,
                    mae_test=mae_test,
                    n_params=0  # Los baselines no tienen parámetros entrenables
                )

    print("\nBaselines evaluados para las 16 combinaciones.")
