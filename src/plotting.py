"""
plotting.py — Gráficas estandarizadas del proyecto
====================================================

UTILIDAD:
    Centraliza TODAS las funciones de visualización para que las gráficas
    del proyecto tengan un formato consistente, independientemente de
    quién las genere. El enunciado pide gráficas muy específicas:
    - Curvas de convergencia de cada entrenamiento.
    - 1 gráfica por cada combinación de ventanas entrada-salida (16 gráficas).
    - 1 gráfica resumen por cada ventana de salida (4 gráficas).
    - Matriz de competición 4x4 con los mejores modelos en test.

OBJETIVO:
    Que cualquier miembro del equipo pueda generar gráficas con una sola
    llamada a función, sin preocuparse por formato, colores o layout.
    Esto ahorra tiempo en la fase final de integración y garantiza
    coherencia visual en la presentación.

USO TÍPICO:
    from src.plotting import plot_training_curves, plot_competition_matrix

    plot_training_curves(history, model_name="LSTM_v1", input_window=30, output_window=5)
    plot_competition_matrix()  # genera la matriz 4x4 automáticamente desde los resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_WINDOWS, OUTPUT_WINDOWS, FIGURES_DIR, TABLES_DIR


# ============================================================================
# CURVAS DE CONVERGENCIA (entrenamiento)
# ============================================================================

def plot_training_curves(history, model_name, input_window, output_window,
                         save=True):
    """
    Genera la gráfica de curvas de entrenamiento (loss train vs val por época).

    El enunciado exige: "Para cada entrenamiento las curvas de entrenamiento
    donde se vea que el modelo ha convergido."

    Parameters
    ----------
    history : keras.callbacks.History
        Objeto history devuelto por model.fit().
    model_name : str
        Nombre del modelo para el título y nombre de archivo.
    input_window : int
        Ventana de entrada usada.
    output_window : int
        Ventana de salida usada.
    save : bool
        Si True, guarda la figura en results/figures/.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history.history['loss'], label='Train Loss (MAE)', linewidth=1.5)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Val Loss (MAE)', linewidth=1.5)

    ax.set_title(f'{model_name} — Convergencia (in={input_window}, out={output_window})')
    ax.set_xlabel('Época')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        filename = f"convergence_{model_name}_in{input_window}_out{output_window}.png"
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
        print(f"Gráfica guardada: {filename}")

    plt.show()
    plt.close(fig)


# ============================================================================
# GRÁFICA POR COMBINACIÓN DE VENTANAS
# ============================================================================

def plot_window_comparison(input_window, output_window, results_df=None,
                           save=True):
    """
    Genera una gráfica comparativa de todos los modelos para una combinación
    específica de ventana entrada-salida.

    El enunciado pide: "Una gráfica de resultados de cada modelo para cada
    combinación de tamaños de ventanas entrada-salida (16 gráficas)."

    Parameters
    ----------
    input_window : int
        Ventana de entrada.
    output_window : int
        Ventana de salida.
    results_df : pd.DataFrame, optional
        DataFrame con resultados. Si None, lo carga del CSV.
    save : bool
        Si True, guarda la figura.
    """
    if results_df is None:
        results_path = os.path.join(TABLES_DIR, 'all_results.csv')
        results_df = pd.read_csv(results_path)

    # Filtrar por combinación de ventanas
    mask = (results_df['input_window'] == input_window) & \
           (results_df['output_window'] == output_window)
    df = results_df[mask].sort_values('mae_test')

    if df.empty:
        print(f"No hay resultados para in={input_window}, out={output_window}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'dense': '#2196F3',
        'recurrent': '#FF9800',
        'convolutional': '#4CAF50',
        'mixed': '#9C27B0'
    }

    bars = ax.barh(
        df['model_name'],
        df['mae_test'],
        color=[colors.get(t, '#757575') for t in df['model_type']]
    )

    ax.set_xlabel('MAE (Test)')
    ax.set_title(f'Comparativa modelos — Entrada={input_window}d, Salida={output_window}d')
    ax.grid(True, axis='x', alpha=0.3)

    # Leyenda por tipo de modelo
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize())
                       for t, c in colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        filename = f"comparison_in{input_window}_out{output_window}.png"
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# ============================================================================
# GRÁFICA RESUMEN POR VENTANA DE SALIDA
# ============================================================================

def plot_output_window_summary(output_window, results_df=None, save=True):
    """
    Genera una gráfica que contiene los resultados de todos los modelos
    agrupados por ventana de entrada, para una ventana de salida fija.

    El enunciado pide: "Una gráfica que contenga todos los resultados
    para cada tamaño de ventana de salida."

    Parameters
    ----------
    output_window : int
        Ventana de salida fija (1, 5, 30, o 90).
    results_df : pd.DataFrame, optional
        DataFrame con resultados.
    save : bool
        Si True, guarda la figura.
    """
    if results_df is None:
        results_path = os.path.join(TABLES_DIR, 'all_results.csv')
        results_df = pd.read_csv(results_path)

    mask = results_df['output_window'] == output_window
    df = results_df[mask]

    if df.empty:
        print(f"No hay resultados para output_window={output_window}")
        return

    fig, axes = plt.subplots(1, len(INPUT_WINDOWS), figsize=(20, 6),
                              sharey=True)
    fig.suptitle(f'Resumen — Ventana de salida = {output_window} días',
                 fontsize=14, fontweight='bold')

    colors = {
        'dense': '#2196F3',
        'recurrent': '#FF9800',
        'convolutional': '#4CAF50',
        'mixed': '#9C27B0'
    }

    for idx, iw in enumerate(INPUT_WINDOWS):
        ax = axes[idx]
        sub = df[df['input_window'] == iw].sort_values('mae_test')

        if not sub.empty:
            ax.barh(
                sub['model_name'],
                sub['mae_test'],
                color=[colors.get(t, '#757575') for t in sub['model_type']]
            )

        ax.set_title(f'Entrada = {iw}d')
        ax.grid(True, axis='x', alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Modelo')

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        filename = f"summary_out{output_window}.png"
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# ============================================================================
# MATRIZ DE COMPETICIÓN 4x4
# ============================================================================

def plot_competition_matrix(results_df=None, save=True):
    """
    Genera la matriz 4x4 (ventanas entrada x ventanas salida) con el
    mejor MAE en test para cada combinación.

    El enunciado pide: "Resultado de la competición: una matriz de tamaño
    #ventanas de entrada x #ventanas de salida con los resultados en test
    de los mejores modelos para cada combinación."

    Parameters
    ----------
    results_df : pd.DataFrame, optional
        DataFrame con todos los resultados.
    save : bool
        Si True, guarda la figura y el CSV.
    """
    if results_df is None:
        results_path = os.path.join(TABLES_DIR, 'all_results.csv')
        results_df = pd.read_csv(results_path)

    # Construir la matriz: mejor MAE test por combinación
    matrix = pd.DataFrame(index=INPUT_WINDOWS, columns=OUTPUT_WINDOWS, dtype=float)
    best_models = pd.DataFrame(index=INPUT_WINDOWS, columns=OUTPUT_WINDOWS, dtype=str)

    for iw in INPUT_WINDOWS:
        for ow in OUTPUT_WINDOWS:
            mask = (results_df['input_window'] == iw) & \
                   (results_df['output_window'] == ow)
            sub = results_df[mask]
            if not sub.empty:
                best_idx = sub['mae_test'].idxmin()
                matrix.loc[iw, ow] = sub.loc[best_idx, 'mae_test']
                best_models.loc[iw, ow] = sub.loc[best_idx, 'model_name']

    # Visualización
    fig, ax = plt.subplots(figsize=(10, 8))

    matrix_values = matrix.values.astype(float)
    im = ax.imshow(matrix_values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(OUTPUT_WINDOWS)))
    ax.set_xticklabels([f'{ow}d' for ow in OUTPUT_WINDOWS])
    ax.set_yticks(range(len(INPUT_WINDOWS)))
    ax.set_yticklabels([f'{iw}d' for iw in INPUT_WINDOWS])
    ax.set_xlabel('Ventana de salida')
    ax.set_ylabel('Ventana de entrada')
    ax.set_title('Matriz de Competición — Mejor MAE en Test')

    # Anotar cada celda con MAE y nombre del modelo
    for i in range(len(INPUT_WINDOWS)):
        for j in range(len(OUTPUT_WINDOWS)):
            mae_val = matrix_values[i, j]
            model = best_models.iloc[i, j]
            if not np.isnan(mae_val):
                ax.text(j, i, f'{mae_val:.5f}\n({model})',
                        ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, label='MAE Test')
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'competition_matrix.png'),
                    dpi=150, bbox_inches='tight')
        matrix.to_csv(os.path.join(TABLES_DIR, 'competition_matrix.csv'))
        best_models.to_csv(os.path.join(TABLES_DIR, 'competition_best_models.csv'))
        print("Matriz de competición guardada.")

    plt.show()
    plt.close(fig)


# ============================================================================
# GENERAR TODAS LAS GRÁFICAS DEL ENUNCIADO
# ============================================================================

def generate_all_plots():
    """
    Genera todas las gráficas requeridas por el enunciado de una sola vez:
    - 16 gráficas de comparación (una por combinación de ventanas)
    - 4 gráficas resumen (una por ventana de salida)
    - 1 matriz de competición 4x4
    """
    results_path = os.path.join(TABLES_DIR, 'all_results.csv')
    results_df = pd.read_csv(results_path)

    print("=" * 60)
    print("Generando 16 gráficas de comparación por combinación...")
    print("=" * 60)
    for iw in INPUT_WINDOWS:
        for ow in OUTPUT_WINDOWS:
            plot_window_comparison(iw, ow, results_df)

    print("\n" + "=" * 60)
    print("Generando 4 gráficas resumen por ventana de salida...")
    print("=" * 60)
    for ow in OUTPUT_WINDOWS:
        plot_output_window_summary(ow, results_df)

    print("\n" + "=" * 60)
    print("Generando matriz de competición...")
    print("=" * 60)
    plot_competition_matrix(results_df)

    print("\n¡Todas las gráficas generadas!")
