"""
portfolio.py — Construcción y comparación de carteras (parte de investigación)
===============================================================================

UTILIDAD:
    Implementa la parte de investigación del enunciado donde hay que
    construir dos carteras y comparar su rendimiento en 2025:
    1. Una cartera que NO usa predicciones del modelo.
    2. Una cartera que SÍ usa predicciones del modelo.

    El enunciado dice: "Dado el mejor modelo para una ventana de salida
    de 90 días, implementa dos carteras: una que no use las predicciones
    y una que sí. Compara los resultados de las carteras para el 2025."

OBJETIVO:
    Demostrar si las predicciones de la red neuronal aportan valor real
    en la gestión de una cartera frente a una estrategia pasiva.

NOTA:
    Este módulo se desarrollará en la fase de investigación, después de
    completar la competición. Se incluye en la infraestructura común
    desde el principio para que la estructura del repo esté completa
    y el código de carteras sea consistente entre los miembros del equipo.

USO TÍPICO:
    from src.portfolio import run_portfolio_comparison

    results = run_portfolio_comparison(model, returns_2025)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, FIGURES_DIR


# ============================================================================
# CARTERA SIN PREDICCIONES (Equal Weight / Buy & Hold)
# ============================================================================

def portfolio_equal_weight(returns):
    """
    Cartera equiponderada: asigna el mismo peso a los 23 activos
    y mantiene la posición durante todo el periodo (buy & hold).

    Parameters
    ----------
    returns : pd.DataFrame
        Log-returns diarios de los activos para el periodo de evaluación.

    Returns
    -------
    pd.Series
        Retorno acumulado diario de la cartera.
    """
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    portfolio_returns = returns @ weights
    cumulative = (1 + portfolio_returns).cumprod()
    return cumulative


# ============================================================================
# CARTERA CON PREDICCIONES
# ============================================================================

def portfolio_with_predictions(returns, predictions, rebalance_days=90):
    """
    Cartera que usa las predicciones del modelo para asignar pesos.

    Estrategia: sobrepondera activos con mayor return predicho,
    infrapondera activos con menor return predicho.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-returns diarios reales del periodo de evaluación.
    predictions : np.ndarray
        Predicciones del modelo para cada ventana de 90 días.
    rebalance_days : int
        Frecuencia de rebalanceo en días.

    Returns
    -------
    pd.Series
        Retorno acumulado diario de la cartera.
    """
    # TODO: Implementar en la fase de investigación.
    # La lógica de asignación de pesos dependerá del análisis
    # de las predicciones del mejor modelo.
    raise NotImplementedError(
        "Implementar en la fase de investigación, una vez seleccionado "
        "el mejor modelo para output_window=90."
    )


# ============================================================================
# COMPARACIÓN DE CARTERAS
# ============================================================================

def plot_portfolio_comparison(cumulative_passive, cumulative_active,
                               save=True):
    """
    Compara visualmente el rendimiento de ambas carteras en 2025.

    Parameters
    ----------
    cumulative_passive : pd.Series
        Retorno acumulado de la cartera sin predicciones.
    cumulative_active : pd.Series
        Retorno acumulado de la cartera con predicciones.
    save : bool
        Si True, guarda la figura.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(cumulative_passive.index, cumulative_passive.values,
            label='Sin predicciones (Equal Weight)', linewidth=1.5)
    ax.plot(cumulative_active.index, cumulative_active.values,
            label='Con predicciones (Modelo NN)', linewidth=1.5)

    ax.set_title('Comparación de Carteras — 2025')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor acumulado (base = 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'portfolio_comparison_2025.png'),
                    dpi=150, bbox_inches='tight')
        print("Gráfica de carteras guardada.")

    plt.show()
    plt.close(fig)
