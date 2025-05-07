import numpy as np
import cmath
import time
import matplotlib.pyplot as plt

# ==============================
# Función de Zero Padding
# ==============================
def zero_padding(senal, largo_salida):
    ceros_a_agregar = largo_salida - len(senal)
    return np.concatenate((senal, np.zeros(ceros_a_agregar, dtype=np.complex128)))

# ==============================
# Convolución Circular Iterativa (SCC)
# ==============================
def scc(x, h):
    largoX = len(x)
    largoH = len(h)
    N = largoX + largoH - 1

    x_padded = zero_padding(x, N)
    h_padded = zero_padding(h, N)

    y = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            y[n] += x_padded[k] * h_padded[(n - k) % N]  # Fórmula de CCS

    return y

# ==============================
# FFT con Divide y Conquistar
# ==============================
def fft(x):
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
# IFFT con Divide y Conquistar
# ==============================
def ifft(X):
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
    return x / 2  # Divide en cada nivel

# ==============================
# CCS usando FFT/IFFT
# ==============================
def ccs_fft(x, h):
    largoX = len(x)
    largoH = len(h)
    N = largoX + largoH - 1
    m = int(np.ceil(np.log2(N)))  # Redondear hacia arriba
    N_power2 = 2 ** m  # Asegurar que N sea potencia de 2

    x_padded = zero_padding(x, N_power2)
    h_padded = zero_padding(h, N_power2)

    X = fft(x_padded)
    H = fft(h_padded)

    Y = np.array([X[i] * H[i] for i in range(N_power2)], dtype=np.complex128)

    y = ifft(Y)
    return y[:N]

# ==============================
# Comparación de tiempos y errores
# ==============================
def implementar_ccs():
    print("Comparación de tiempos con señales aleatorias entre -1 y 1:\n")

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

        tamanos.append(N)
        tiempos_iter.append(t_iter)
        tiempos_fft.append(t_fft)
        errores.append(error_prom)

        print(f"N = {N}: Iterativo = {t_iter:.6f}s | FFT = {t_fft:.6f}s | Error = {error_prom:.2e}")

    # Gráfico de tiempos
    plt.figure(figsize=(12, 6))
    plt.plot(tamanos, tiempos_iter, marker='o', linestyle='-', color='blue', label='Iterativo (SCC) ~ O(N²)')
    plt.plot(tamanos, tiempos_fft, marker='s', linestyle='--', color='orange', label='FFT ~ O(N log N)')
    plt.xlabel('Tamaño de señal (N)')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempo: Método Iterativo vs FFT')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("comparacion_tiempos.png", dpi=300)
    plt.show()

    # Gráfico de error
    plt.figure(figsize=(12, 6))
    plt.plot(tamanos, errores, marker='^', color='red', label='Error promedio')
    plt.xlabel('Tamaño de señal (N)')
    plt.ylabel('Error absoluto máximo promedio')
    plt.title('Error promedio entre métodos: SCC vs FFT')
    plt.grid(True, which='both')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("error_promedio.png")
    plt.show()

# Ejecutar
if __name__ == "__main__":
    implementar_ccs()
