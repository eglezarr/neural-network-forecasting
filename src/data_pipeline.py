"""
data_pipeline.py — Carga de datos y generación de ventanas temporales
======================================================================

UTILIDAD:
    Encapsula todo el código del notebook del profesor en funciones
    reutilizables. De esta forma, cualquier miembro del equipo puede
    obtener los datos listos para entrenar con dos líneas de código,
    sin tener que copiar/pegar celdas del notebook original.

OBJETIVO:
    Garantizar que los 3 miembros del equipo usen EXACTAMENTE los mismos
    datos, las mismas transformaciones y el mismo split train/test.
    Si cada uno copia el código del profesor por su cuenta, hay riesgo
    de pequeñas diferencias que harían los resultados incomparables.

ORIGEN DEL CÓDIGO:
    - La función create_time_series_data() es EXACTA del profesor.
      No se ha modificado para mantener compatibilidad con la competición.
    - Las funciones load_data() y get_train_test() son wrappers nuestros
      que encapsulan el flujo del notebook en una interfaz limpia.

USO TÍPICO:
    from src.data_pipeline import load_data, get_train_test

    returns = load_data()
    X_train, X_test, y_train, y_test = get_train_test(returns, input_window=30, output_window=5)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, START_DATE, END_DATE, RANDOM_SEED, TEST_SIZE


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_data():
    """
    Descarga precios de cierre de los 23 tickers del S&P 500 y calcula
    log-returns diarios.

    Replica exactamente el código del notebook del profesor:
        precios_close = yf.download(tickers, start='1945-01-01', auto_adjust=True)['Close']
        returns = np.log(precios_close).diff().dropna()

    Se usa END_DATE fija para garantizar que todos los miembros del equipo
    trabajen con exactamente los mismos datos independientemente de cuándo
    ejecuten el código.

    Returns
    -------
    returns : pd.DataFrame
        Log-returns diarios, shape esperado ~(16181, 23).
    """
    precios_close = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True
    )['Close']

    precios_close.dropna(axis=1, inplace=True)

    returns = np.log(precios_close).diff().dropna()

    print(f"Datos cargados: {returns.shape[0]} días, {returns.shape[1]} activos")
    print(f"Rango: {returns.index[0].strftime('%Y-%m-%d')} → {returns.index[-1].strftime('%Y-%m-%d')}")

    return returns


# ============================================================================
# GENERACIÓN DE VENTANAS — CÓDIGO EXACTO DEL PROFESOR (NO MODIFICAR)
# ============================================================================

def create_time_series_data(data, input_window_size, output_window_size):
    """
    Genera secuencias de entrada y promedios de salida para datos de series temporales.

    NOTA: Esta función es EXACTA del notebook del profesor.
    No se ha modificado para mantener compatibilidad con la competición.

    Args:
        data (pd.DataFrame o np.array): Los datos de la serie temporal.
        input_window_size (int): El número de pasos de tiempo para la secuencia de entrada (X).
        output_window_size (int): El número de pasos de tiempo para calcular el promedio de la salida (y).

    Returns:
        tuple: (X, y) donde X son las secuencias de entrada y y son los promedios de salida.
               X tendrá la forma (num_samples, input_window_size, num_features).
               y tendrá la forma (num_samples, num_features).
    """
    X, y = [], []
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    num_features = data_array.shape[1]

    for i in range(len(data_array) - input_window_size - output_window_size + 1):
        # Secuencia de entrada
        input_sequence = data_array[i : i + input_window_size]
        X.append(input_sequence)

        # Salida: promedio de los siguientes 'output_window_size' pasos
        if output_window_size > 0:
            output_sequence = data_array[
                i + input_window_size : i + input_window_size + output_window_size
            ]
            average_output = np.mean(output_sequence, axis=0)
            y.append(average_output)
        else:
            y.append(data_array[i + input_window_size - 1])

    return np.array(X), np.array(y)


# ============================================================================
# WRAPPER PRINCIPAL — Lo que cada miembro del equipo debe llamar
# ============================================================================

def get_train_test(data, input_window, output_window):
    """
    Pipeline completo: genera ventanas temporales y hace split train/test.

    Combina create_time_series_data() + train_test_split() con los
    parámetros exactos del notebook del profesor (shuffle=False, seed=42).

    Parameters
    ----------
    data : pd.DataFrame
        Log-returns diarios (output de load_data()).
    input_window : int
        Días de historia como entrada. Valores del enunciado: 5, 10, 30, 90.
    output_window : int
        Días a futuro para el promedio de salida. Valores del enunciado: 1, 5, 30, 90.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        X_train shape: (N_train, input_window, 23)
        y_train shape: (N_train, 23)
        X_test shape:  (N_test, input_window, 23)
        y_test shape:  (N_test, 23)
    """
    X, y = create_time_series_data(data, input_window, output_window)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,
        random_state=RANDOM_SEED
    )

    print(f"Ventana entrada={input_window}, salida={output_window}")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


#  Función de pesos de la diferenciación fraccional

# La diferenciación fraccional aplica una suma ponderada del pasado:
#   Δ^d X_t = Σ w_k · X_{t-k}
#
# Los pesos w_k se calculan con la fórmula recursiva:
#   w_0 = 1
#   w_k = -w_{k-1} · (d - k + 1) / k
#
# Propiedades clave:
#   - Con d=1 → pesos = [1, -1, 0, 0, ...] (diferencia clásica)
#   - Con d<1 → los pesos NUNCA llegan a 0, decaen gradualmente
#   - Cuanto menor es d, más lento decaen → más memoria histórica

def get_weights(d, size):
    """
    Calcula los pesos de la diferenciación fraccional.
    d    : orden de diferenciación (float entre 0 y 1)
    size : número de pesos a calcular (longitud de la ventana)
    """
    w = [1.0]  # w_0 = 1 siempre
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w)


#  Función de diferenciación fraccional

# Para cada punto t, multiplicamos los pesos por los precios
# del pasado y los sumamos:
#   Δ^d X_t = w_0·X_t + w_1·X_{t-1} + w_2·X_{t-2} + ...
#
# Parámetro thresh: descartamos pesos cuyo valor absoluto sea
# menor que thresh → evitamos usar un pasado demasiado lejano
# con pesos casi nulos (optimización computacional)

def frac_diff(series, d, thresh=1e-4):
    """
    Aplica diferenciación fraccional a una serie temporal.
    series : pd.Series con los precios
    d      : orden de diferenciación (float entre 0 y 1)
    thresh : umbral para truncar pesos insignificantes
    """
    # Calculamos pesos hasta que sean insignificantes
    pesos = get_weights(d, len(series))

    # Truncamos en el primer peso menor que thresh
    pesos_truncados = np.where(np.abs(pesos) > thresh)[0]
    if len(pesos_truncados) == 0:
        ventana = len(series)
    else:
        ventana = pesos_truncados[-1] + 1

    pesos = pesos[:ventana]

    resultado = []
    # Empezamos desde el índice = ventana para tener suficiente historial
    for t in range(ventana, len(series)):
        # Extraemos los últimos 'ventana' precios (en orden inverso)
        ventana_precios = series.iloc[t - ventana + 1 : t + 1].values[::-1]
        # Producto escalar pesos · precios
        valor = np.dot(pesos, ventana_precios)
        resultado.append(valor)

    # El índice empieza en 'ventana' porque los primeros valores
    # no tienen suficiente historial para calcular
    idx = series.index[ventana:]
    return pd.Series(resultado, index=idx, name=f'frac_diff_d{d}')


def load_data_lopez_de_prado(d=0.4, thres=1e-4):
    """
    Carga los precios de cierre, calcula los log-precios y les aplica
    la diferenciación fraccional de López de Prado para generar los retornos LP.
    """
    # 1. Cargar precios de cierre originales (exactamente como en tu load_data original)
    precios_close = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )['Close']
    precios_close.dropna(axis=1, inplace=True)
    
    # 2. Calcular los log-precios (es vital aplicar FFD sobre el LOG del precio)
    log_precios = np.log(precios_close)
    
    # 3. Aplicar diferenciación fraccional columna por columna
    returns_list = []
    for col in log_precios.columns:
        diff_col = frac_diff(log_precios[col], d, thres)
        returns_list.append(diff_col)
        
    # Re-construimos el DataFrame con las columnas originales y limpiamos NaNs
    returns_lp = pd.concat(returns_list, axis=1)
    returns_lp.columns = log_precios.columns
    returns_lp.dropna(inplace=True)
    
    print(f"Datos de López de Prado cargados: {returns_lp.shape[0]} días, {returns_lp.shape[1]} activos (d={d})")
    print(f"Rango: {returns_lp.index[0].strftime('%Y-%m-%d')} → {returns_lp.index[-1].strftime('%Y-%m-%d')}")
    
    return returns_lp





# Test ADF (Augmented Dickey-Fuller)

# El test ADF mide estadísticamente si una serie es estacionaria.
#
# Hipótesis:
#   H0 (hipótesis nula)      → la serie NO es estacionaria
#   H1 (hipótesis alternativa) → la serie SÍ es estacionaria
#
# Interpretación del p-valor:
#   p-valor < 0.05 → rechazamos H0 → serie estacionaria
#   p-valor > 0.05 → no podemos rechazar H0 → no estacionaria
#
# Nuestro objetivo: encontrar el MÍNIMO d con p-valor < 0.05
# → mínima diferenciación que consigue estacionariedad

from statsmodels.tsa.stattools import adfuller

def test_adf(series, nombre=""):
    """
    Aplica el test ADF a una serie y devuelve los resultados clave.
    """
    resultado = adfuller(series.dropna(), maxlag=1, regression='c', autolag=None)
    return {
        'Serie'       : nombre,
        'ADF Statistic': round(resultado[0], 4),
        'p-valor'     : round(resultado[1], 4),
        'Estacionaria': 'SI' if resultado[1] < 0.05 else 'NO'
        }

