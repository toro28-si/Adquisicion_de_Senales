import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Selecciona el tipo de archivo: "bin" o "csv"
modo = "csv"

# Frecuencia de muestreo usada SOLO para archivos BIN
fs_programada = 5000  # Hz

# Tamaño de la FFT
# Si quieres zero padding, usa un valor mayor que el número
# de muestras. Por ejemplo: 2048, 4096, 8192, etc.
N_FFT = 2048

# Rutas de los archivos
archivo_bin = Path(
    r"C:\Users\aalej\OneDrive\Documents\RecolecciónOchoa\F1200.bin"
)

archivo_csv = Path(
    r"C:\Users\aalej\OneDrive\Documents\RecolecciónOchoa\F1100.CSV"
)


# ============================================================
# LECTURA DE DATOS
# ============================================================

if modo == "bin":

    # Lee datos binarios de tipo uint16_t
    # Debe coincidir con el tipo usado en Arduino
    muestras_adc = np.fromfile(archivo_bin, dtype=np.uint16)

    if len(muestras_adc) == 0:
        raise ValueError("El archivo BIN no contiene muestras.")

    # Para BIN usamos la frecuencia programada
    fs = fs_programada

    # Generamos el vector de tiempo
    tiempo = np.arange(len(muestras_adc)) / fs

    nombre = "Adquisición binaria"


elif modo == "csv":

    # Lee las columnas:
    # Columna 0 -> número de muestra
    # Columna 1 -> tiempo en microsegundos
    # Columna 2 -> amplitud ADC
    datos = np.loadtxt(
        archivo_csv,
        delimiter=",",
        skiprows=1
    )

    muestra = datos[:, 0]
    tiempo_us = datos[:, 1]
    muestras_adc = datos[:, 2]

    if len(muestras_adc) == 0:
        raise ValueError("El archivo CSV no contiene muestras.")

    # Convierte microsegundos a segundos
    tiempo = tiempo_us / 1_000_000

    # Calcula la frecuencia de muestreo REAL
    fs = 1 / np.mean(np.diff(tiempo))

    nombre = "Adquisición CSV"

else:
    raise ValueError('modo debe ser "bin" o "csv"')


# ============================================================
# CONVERSIÓN ADC -> VOLTAJE
# ============================================================

voltaje = muestras_adc * 5.0 / 1023.0


# ============================================================
# INFORMACIÓN DE LA ADQUISICIÓN
# ============================================================

print("=" * 50)
print("ARCHIVO:", nombre)
print("Número de muestras:", len(muestras_adc))
print("Duración:", len(muestras_adc) / fs, "s")
print("Frecuencia de muestreo:", fs, "Hz")
print("=" * 50)


# ============================================================
# GRÁFICA DE LA SEÑAL EN EL TIEMPO
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(
    tiempo,
    voltaje,
    linewidth=0.8,
    marker="o",
    markersize=2
)

plt.title(nombre + " — Señal adquirida")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# ELIMINAR COMPONENTE DC
# ============================================================

senal = voltaje - np.mean(voltaje)


# ============================================================
# FFT CON ZERO PADDING
# ============================================================

# Aseguramos que la FFT no sea menor que la señal
if N_FFT < len(senal):
    N_FFT = len(senal)

fft = np.fft.rfft(
    senal,
    n=N_FFT
)

frecuencias = np.fft.rfftfreq(
    N_FFT,
    d=1 / fs
)

magnitud = np.abs(fft) / len(senal)


# ============================================================
# FRECUENCIA DOMINANTE
# ============================================================

# Se omite la posición 0 porque corresponde a DC
indice_pico = np.argmax(magnitud[1:]) + 1

frecuencia_senal = frecuencias[indice_pico]

print(
    "Frecuencia dominante:",
    frecuencia_senal,
    "Hz"
)


# ============================================================
# GRÁFICA DEL ESPECTRO FFT
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(
    frecuencias,
    magnitud,
    linewidth=0.8
)

# Marca la frecuencia dominante
plt.axvline(
    frecuencia_senal,
    linestyle="--",
    label=f"Pico: {frecuencia_senal:.2f} Hz"
)

plt.title(nombre + " — Espectro FFT")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")

# Límite de Nyquist
plt.xlim(0, fs / 2)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()