import numpy as np
import cmath
import time
import matplotlib.pyplot as plt

# ==============================
# Función de Zero Padding
# ==============================
def zero_padding(senal, largo_salida):
    """ Agrega ceros al final de la señal hasta alcanzar el largo especificado """
    ceros_a_agregar = largo_salida - len(senal)
    return np.concatenate((senal, np.zeros(ceros_a_agregar, dtype=np.complex128)))


# ==============================
# SCC - Convolución Circular Iterativa (Retorna Reales)
# ==============================
def scc(x, h):
    """ Implementación iterativa de la Suma de Convolución Circular """
    largoX = len(x)
    largoH = len(h)
    N = largoX + largoH - 1

    x_padded = zero_padding(x, N)
    h_padded = zero_padding(h, N)

    y = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            y[n] += x_padded[k] * h_padded[(n - k) % N]

    return np.real(y)  # Retornar solo parte real


# ==============================
# FFT - Divide y Conquistar
# ==============================
def fft(x):
    """ Implementación recursiva de la FFT """
    N = len(x)
    if N == 1:
        return x

    pares = fft(x[::2])
    impares = fft(x[1::2])

    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        factor = cmath.exp(-2j * np.pi * k / N)
        X[k] = pares[k] + factor * impares[k]
        X[k + N // 2] = pares[k] - factor * impares[k]

    return X


# ==============================
# IFFT - Divide y Conquistar (Retorna Reales)
# ==============================
def ifft(X):
    """ Implementación recursiva de la IFFT """
    N = len(X)
    if N == 1:
        return X

    pares = ifft(X[::2])
    impares = ifft(X[1::2])

    x = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        factor = cmath.exp(2j * np.pi * k / N)
        x[k] = pares[k] + factor * impares[k]
        x[k + N // 2] = pares[k] - factor * impares[k]

    return x / 2  # Dividir por 2 en cada nivel de recursión


# ==============================
# CCS FFT - Convolución Circular con FFT/IFFT
# ==============================
def ccs_fft(x, h):
    """ Implementación de CCS usando FFT e IFFT """
    largoX = len(x)
    largoH = len(h)
    N = largoX + largoH - 1
    m = int(np.ceil(np.log2(N)))
    N_power2 = 2 ** m

    # Zero Padding
    x_padded = zero_padding(x, N_power2)
    h_padded = zero_padding(h, N_power2)

    # FFT
    X = fft(x_padded)
    H = fft(h_padded)

    # Producto en el dominio de la frecuencia
    Y = X * H

    # IFFT y retorno de valores reales
    y = ifft(Y)
    return np.real(y[:N])


# ==============================
# Comparación de tiempos y errores
# ==============================
def implementar_ccs():
    """ Función para comparar tiempos de ejecución y errores """
    tamanos = []
    tiempos_iter = []
    tiempos_fft = []
    errores = []

    for m in range(1, 13):  # Desde 2^1 hasta 2^12
        N = 2**m
        x = np.random.uniform(-1, 1, N).astype(np.complex128)
        h = np.random.uniform(-1, 1, N).astype(np.complex128)

        rep = 10
        t_iter = 0
        t_fft = 0
        error = 0

        for _ in range(rep):
            # SCC iterativo
            inicio = time.perf_counter()
            y_iter = scc(x, h)
            fin = time.perf_counter()
            t_iter += (fin - inicio)

            # CCS con FFT
            inicio = time.perf_counter()
            y_fft = ccs_fft(x, h)
            fin = time.perf_counter()
            t_fft += (fin - inicio)

            # Error absoluto máximo
            error += np.max(np.abs(y_iter - y_fft))

        # Promedios
        t_iter /= rep
        t_fft /= rep
        error_prom = error / rep

        # Almacenar resultados
        tamanos.append(N)
        tiempos_iter.append(t_iter)
        tiempos_fft.append(t_fft)
        errores.append(error_prom)

        print(f"N = {N}: Iterativo = {t_iter:.6f}s | FFT = {t_fft:.6f}s | Error = {error_prom:.2e}")

    # Gráfico de tiempos de ejecución
    plt.plot(tamanos, tiempos_iter, label="Iterativo")
    plt.plot(tamanos, tiempos_fft, label="FFT")
    plt.xlabel("Tamaño de señal (N)")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Tiempo de ejecución: Iterativo vs FFT")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de error
    plt.plot(tamanos, errores, label="Error absoluto máximo", color="red")
    plt.xlabel("Tamaño de señal (N)")
    plt.ylabel("Error")
    plt.title("Error entre métodos")
    plt.grid(True)
    plt.legend()
    plt.show()


# ==============================
# Ejecución principal
# ==============================
if __name__ == "__main__":
    implementar_ccs()

