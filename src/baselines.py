"""
baselines.py — Modelos de referencia (Buy & Hold, Naive, Zero)
===============================================================

UTILIDAD:
    Proporciona modelos simples que sirven como SUELO MÍNIMO de rendimiento.
    Cualquier red neuronal que no bata a estos modelos no está aportando
    valor. El enunciado lo pide explícitamente:
    "Se deben añadir modelos simples, por ejemplo Buy and Hold."

OBJETIVO:
    1. Buy & Hold: predice la media histórica incondicional de los
       log-returns del conjunto de entrenamiento. Simula mantener la
       posición asumiendo que el futuro se comportará como el pasado
       en promedio.
    2. Naive (Last Value): predice que el futuro será igual al último
       valor observado en la ventana de entrada. Es el benchmark más
       básico en series temporales.
    3. Zero Prediction: predice que el log-return futuro promedio será
       cero. Justificación: los log-returns diarios tienen media cercana
       a cero, lo que convierte a este modelo en un baseline
       sorprendentemente difícil de batir en horizontes cortos.

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
# BASELINE 1: BUY & HOLD (media histórica incondicional)
# ============================================================================

def predict_buy_and_hold(X, y_train):
    """
    Predice que el promedio futuro de los log-returns será igual a la
    media histórica del conjunto de entrenamiento.

    Lógica: Buy & Hold asume que el mercado se comportará en el futuro
    como lo ha hecho históricamente en promedio. La predicción es la
    misma para todas las muestras — la media incondicional.

    Parameters
    ----------
    X : np.ndarray
        Secuencias de entrada, shape (N, input_window, 23).
        No se usa para la predicción, pero se incluye para mantener
        la interfaz consistente y determinar el número de muestras.
    y_train : np.ndarray
        Valores de salida del conjunto de entrenamiento, shape (N_train, 23).
        Se usa para calcular la media histórica.

    Returns
    -------
    np.ndarray
        Predicción constante (media histórica), shape (N, 23).
    """
    historical_mean = np.mean(y_train, axis=0)  # Media por activo (23,)
    return np.tile(historical_mean, (X.shape[0], 1))  # Repetir para cada muestra


# ============================================================================
# BASELINE 2: NAIVE (Last Value)
# ============================================================================

def predict_naive(X):
    """
    Predice que el promedio futuro de los log-returns será igual al
    último valor de la ventana de entrada.

    Lógica: asume que mañana será como hoy. Es el benchmark más
    básico en series temporales.

    Parameters
    ----------
    X : np.ndarray
        Secuencias de entrada, shape (N, input_window, 23).

    Returns
    -------
    np.ndarray
        Último timestep de cada secuencia, shape (N, 23).
    """
    return X[:, -1, :]


# ============================================================================
# BASELINE 3: ZERO PREDICTION
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
    print("=" * 60)
    print("Evaluando baselines para todas las combinaciones de ventanas...")
    print("=" * 60)

    for iw in INPUT_WINDOWS:
        for ow in OUTPUT_WINDOWS:
            X_train, X_test, y_train, y_test = get_train_test(returns, iw, ow)

            # --- Buy & Hold (necesita y_train para la media histórica) ---
            y_pred_train_bh = predict_buy_and_hold(X_train, y_train)
            y_pred_test_bh = predict_buy_and_hold(X_test, y_train)

            save_results(
                model_name="Buy_and_Hold",
                model_type="baseline",
                input_window=iw,
                output_window=ow,
                mae_train=compute_mae(y_train, y_pred_train_bh),
                mae_test=compute_mae(y_test, y_pred_test_bh),
                n_params=0
            )

            # --- Naive y Zero (no necesitan y_train) ---
            baselines_simple = {
                'Naive_LastValue': predict_naive,
                'Zero_Prediction': predict_zero,
            }

            for name, predict_fn in baselines_simple.items():
                y_pred_train = predict_fn(X_train)
                y_pred_test = predict_fn(X_test)

                save_results(
                    model_name=name,
                    model_type="baseline",
                    input_window=iw,
                    output_window=ow,
                    mae_train=compute_mae(y_train, y_pred_train),
                    mae_test=compute_mae(y_test, y_pred_test),
                    n_params=0
                )

    print("\nBaselines evaluados para las 16 combinaciones.")
