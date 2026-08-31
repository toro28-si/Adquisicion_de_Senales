"""
PRÁCTICA 1: ADQUISICIÓN DE SEÑALES
ANÁLISIS Y VISUALIZACIÓN DE DATOS

Este script genera todas las gráficas necesarias para el reporte:
1. Señal en el tiempo (500 Hz)
2. Comparativa de señales (500, 1000, 1200, 1500 Hz)
3. Análisis de offset
4. Impacto del offset en FFT
5. Espectros FFT en escala lineal
6. Espectros FFT en escala dB
7. Comparativa CSV vs BIN
8. Detección de frecuencia dominante
9. Puntos por ciclo vs frecuencia
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

FS = 5000              # Frecuencia de muestreo (Hz)
N_FFT = 2048           # Tamaño de la FFT
V_REF = 5.0            # Voltaje de referencia del ADC
ADC_BITS = 1023        # Resolución del ADC (10 bits)
V_TEORICO = 2.5        # Offset teórico (V)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_datos(archivo):
    """Carga un archivo CSV y devuelve el voltaje y tiempo."""
    datos = np.loadtxt(archivo, delimiter=',', skiprows=1)
    tiempo = datos[:, 1] / 1_000_000  # Convertir a segundos
    voltaje = datos[:, 2] * V_REF / ADC_BITS
    return tiempo, voltaje

def calcular_fft(voltaje, fs=FS, n_fft=N_FFT):
    """Calcula la FFT de una señal y devuelve frecuencias y magnitud."""
    senal = voltaje - np.mean(voltaje)
    fft = np.fft.rfft(senal, n=n_fft)
    frecuencias = np.fft.rfftfreq(n_fft, d=1/fs)
    magnitud = np.abs(fft) / len(senal)
    return frecuencias, magnitud

def calcular_fft_db(voltaje, fs=FS, n_fft=N_FFT):
    """Calcula la FFT en dB."""
    frecuencias, magnitud = calcular_fft(voltaje, fs, n_fft)
    magnitud_db = 20 * np.log10(magnitud + 1e-12)
    return frecuencias, magnitud_db

# ============================================================
# 1. SEÑAL EN EL TIEMPO (500 Hz)
# ============================================================

print("\n" + "="*60)
print("1. SEÑAL EN EL TIEMPO - 500 Hz")
print("="*60)

tiempo, voltaje = cargar_datos('F500.CSV')
v_promedio = np.mean(voltaje)
v_pp = np.ptp(voltaje)

plt.figure(figsize=(12, 5))
plt.plot(tiempo, voltaje, 'b-', linewidth=1.5, marker='o', markersize=4)
plt.axhline(y=v_promedio, color='r', linestyle='--', alpha=0.7, 
            label=f'Offset real = {v_promedio:.3f} V')
plt.axhline(y=2.5, color='gray', linestyle=':', alpha=0.5, 
            label='Offset teórico = 2.500 V')
plt.text(0.001, v_promedio + 0.1, f'V_prom = {v_promedio:.3f}V', 
         fontsize=10, color='red')
plt.text(0.001, 2.5 + 0.1, f'V_teo = 2.500V', fontsize=10, color='gray')
plt.title(f'Señal de 500 Hz - Offset real: {v_promedio:.3f}V, Vpp: {v_pp:.3f}V', 
          fontsize=14)
plt.xlabel('Tiempo (s)', fontsize=12)
plt.ylabel('Voltaje (V)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('grafica1_tiempo_offset_real.png', dpi=300)
plt.show()
print("✅ Gráfica 1 guardada: grafica1_tiempo_offset_real.png")

# ============================================================
# 2. COMPARATIVA DE SEÑALES
# ============================================================

print("\n" + "="*60)
print("2. COMPARATIVA DE SEÑALES")
print("="*60)

archivos_comp = ['F500.CSV', 'F1000.CSV', 'F1200.CSV', 'F1500.CSV']
titulos_comp = ['500 Hz - 10 pts/ciclo', 
                '1000 Hz - 5 pts/ciclo',
                '1200 Hz - 4.17 pts/ciclo',
                '1500 Hz - 3.33 pts/ciclo']

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for idx, archivo in enumerate(archivos_comp):
    tiempo, voltaje = cargar_datos(archivo)
    v_promedio = np.mean(voltaje)
    v_pp = np.ptp(voltaje)
    
    ax = axes[idx]
    ax.plot(tiempo, voltaje, 'b-', linewidth=1.5, marker='o', markersize=4)
    ax.axhline(y=v_promedio, color='r', linestyle='--', alpha=0.7)
    ax.text(0.001, v_promedio + 0.1, f'V_prom = {v_promedio:.3f}V', 
            fontsize=9, color='red')
    ax.set_title(f'{titulos_comp[idx]} - Vpp: {v_pp:.3f}V', fontsize=12)
    ax.set_ylabel('Voltaje (V)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 5.5)

axes[-1].set_xlabel('Tiempo (s)', fontsize=12)
plt.tight_layout()
plt.savefig('grafica2_comparativa_offset_real.png', dpi=300)
plt.show()
print("✅ Gráfica 2 guardada: grafica2_comparativa_offset_real.png")

# ============================================================
# 3. ANÁLISIS DE OFFSET
# ============================================================

print("\n" + "="*60)
print("3. ANÁLISIS DE OFFSET")
print("="*60)

archivos = sorted(Path('.').glob('F*.CSV'))
resultados_offset = []

print(f"\n{'Frecuencia (Hz)':>14} | {'Offset (V)':>12} | {'Error (mV)':>12} | {'Vpp (V)':>8}")
print("-" * 60)

for archivo in archivos:
    _, voltaje = cargar_datos(archivo)
    v_prom = np.mean(voltaje)
    v_pico = np.ptp(voltaje)
    error_mv = (v_prom - V_TEORICO) * 1000
    f = int(archivo.stem[1:])
    
    resultados_offset.append({
        'Frecuencia (Hz)': f,
        'Offset real (V)': v_prom,
        'Error offset (mV)': error_mv,
        'Vpp (V)': v_pico
    })
    print(f"{f:14d} | {v_prom:12.3f} | {error_mv:12.1f} | {v_pico:8.2f}")

df_offset = pd.DataFrame(resultados_offset)
media_error = np.mean(df_offset['Error offset (mV)'])
std_error = np.std(df_offset['Error offset (mV)'])

print("-" * 60)
print(f"{'PROMEDIO':>14} | {np.mean(df_offset['Offset real (V)']):12.3f} | {media_error:12.1f} | {np.mean(df_offset['Vpp (V)']):8.2f}")
print(f"{'DESV. EST.':>14} | {np.std(df_offset['Offset real (V)']):12.3f} | {std_error:12.1f} | {np.std(df_offset['Vpp (V)']):8.2f}")

# Gráfica de offset
plt.figure(figsize=(12, 5))
plt.bar(df_offset['Frecuencia (Hz)'], df_offset['Error offset (mV)'], 
        width=60, color='steelblue', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Offset teórico (0 mV)')
plt.axhline(y=media_error, color='g', linestyle='-.', alpha=0.7, linewidth=1.5, 
            label=f'Promedio: {media_error:.1f} mV')
plt.xlabel('Frecuencia (Hz)', fontsize=12)
plt.ylabel('Error de offset (mV)', fontsize=12)
plt.title('Desviación del offset vs frecuencia', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

for i, row in df_offset.iterrows():
    plt.text(row['Frecuencia (Hz)'], row['Error offset (mV)'] - 1.5, 
             f'{row["Error offset (mV)"]:.0f}', ha='center', va='top',
             fontsize=8, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('offset_analysis.png', dpi=300)
plt.show()
print("✅ Gráfica 3 guardada: offset_analysis.png")

# ============================================================
# 4. IMPACTO DEL OFFSET EN FFT
# ============================================================

print("\n" + "="*60)
print("4. IMPACTO DEL OFFSET EN FFT")
print("="*60)

_, voltaje = cargar_datos('F500.CSV')
v_promedio = np.mean(voltaje)

# Caso 1: Offset teórico
senal_teorica = voltaje - V_TEORICO
fft_teorica = np.fft.rfft(senal_teorica, n=N_FFT)
frecuencias = np.fft.rfftfreq(N_FFT, d=1/FS)
magnitud_teorica = np.abs(fft_teorica) / len(senal_teorica)

# Caso 2: Offset real
senal_real = voltaje - v_promedio
fft_real = np.fft.rfft(senal_real, n=N_FFT)
magnitud_real = np.abs(fft_real) / len(senal_real)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Señal con offset teórico
axes[0, 0].plot(tiempo, voltaje, 'b-', linewidth=1)
axes[0, 0].axhline(y=V_TEORICO, color='r', linestyle='--', alpha=0.7)
axes[0, 0].set_title('Señal con offset teórico (2.5V)')
axes[0, 0].set_ylabel('Voltaje (V)')
axes[0, 0].grid(True, alpha=0.3)

# Señal con offset real
axes[0, 1].plot(tiempo, voltaje, 'b-', linewidth=1)
axes[0, 1].axhline(y=v_promedio, color='g', linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=V_TEORICO, color='r', linestyle=':', alpha=0.3)
axes[0, 1].set_title(f'Señal con offset real ({v_promedio:.3f}V)')
axes[0, 1].set_ylabel('Voltaje (V)')
axes[0, 1].grid(True, alpha=0.3)

# FFT con offset teórico
axes[1, 0].plot(frecuencias[:100], magnitud_teorica[:100], 'b-', linewidth=1.5)
axes[1, 0].set_title(f'FFT con offset teórico\nDC = {magnitud_teorica[0]:.4f}')
axes[1, 0].set_xlabel('Frecuencia (Hz)')
axes[1, 0].set_ylabel('Magnitud')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 100)

# FFT con offset real
axes[1, 1].plot(frecuencias[:100], magnitud_real[:100], 'g-', linewidth=1.5)
axes[1, 1].set_title(f'FFT con offset real\nDC = {magnitud_real[0]:.4f}')
axes[1, 1].set_xlabel('Frecuencia (Hz)')
axes[1, 1].set_ylabel('Magnitud')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, 100)

plt.suptitle('Impacto del offset en el análisis espectral', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('offset_impacto_fft.png', dpi=300)
plt.show()
print("✅ Gráfica 4 guardada: offset_impacto_fft.png")

# ============================================================
# 5. ESPECTROS FFT EN ESCALA LINEAL
# ============================================================

print("\n" + "="*60)
print("5. ESPECTROS FFT EN ESCALA LINEAL")
print("="*60)

archivos = sorted(Path('.').glob('F*.CSV'), key=lambda x: int(x.stem[1:]))

n_archivos = len(archivos)
n_cols = 4
n_rows = (n_archivos + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes = axes.flatten()

for idx, archivo in enumerate(archivos):
    _, voltaje = cargar_datos(archivo)
    f_esperada = int(archivo.stem[1:])
    
    frecuencias, magnitud = calcular_fft(voltaje)
    
    ax = axes[idx]
    ax.plot(frecuencias, magnitud, 'b-', linewidth=1.2)
    ax.axvline(f_esperada, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    
    for n in [2, 3, 4]:
        f_arm = n * f_esperada
        if f_arm < 2500:
            ax.axvline(f_arm, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlim(0, 2500)
    ax.grid(True, alpha=0.2)
    
    if idx % n_cols == 0:
        ax.set_ylabel('Magnitud', fontsize=10)
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Frecuencia (Hz)', fontsize=10)
    
    puntos_ciclo = FS / f_esperada
    ax.set_title(f'{f_esperada} Hz ({puntos_ciclo:.1f} pts/ciclo)', 
                 fontsize=11, fontweight='bold')
    ax.text(f_esperada + 30, max(magnitud) * 0.8, f'{f_esperada} Hz', 
            fontsize=8, color='red', fontweight='bold')

for idx in range(len(archivos), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Espectros FFT en escala lineal - Frecuencias ordenadas (500 Hz → 1500 Hz)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('espectros_lineales.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfica 5 guardada: espectros_lineales.png")

# ============================================================
# 6. ESPECTROS FFT EN ESCALA dB
# ============================================================

print("\n" + "="*60)
print("6. ESPECTROS FFT EN ESCALA dB")
print("="*60)

archivos = sorted(Path('.').glob('F*.CSV'), key=lambda x: int(x.stem[1:]))

n_archivos = len(archivos)
n_cols = 4
n_rows = (n_archivos + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes = axes.flatten()

for idx, archivo in enumerate(archivos):
    _, voltaje = cargar_datos(archivo)
    f_esperada = int(archivo.stem[1:])
    
    frecuencias, magnitud_db = calcular_fft_db(voltaje)
    
    ax = axes[idx]
    ax.plot(frecuencias, magnitud_db, 'b-', linewidth=1.2)
    ax.axvline(f_esperada, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    
    for n in [2, 3, 4]:
        f_arm = n * f_esperada
        if f_arm < 2500:
            ax.axvline(f_arm, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlim(0, 2500)
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha=0.2)
    
    if idx % n_cols == 0:
        ax.set_ylabel('Magnitud (dB)', fontsize=10)
    if idx >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Frecuencia (Hz)', fontsize=10)
    
    puntos_ciclo = FS / f_esperada
    ax.set_title(f'{f_esperada} Hz ({puntos_ciclo:.1f} pts/ciclo)', 
                 fontsize=11, fontweight='bold')
    ax.text(f_esperada + 30, -10, f'{f_esperada}', fontsize=8, color='red', 
            fontweight='bold')

for idx in range(len(archivos), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Espectros FFT en escala dB - Frecuencias ordenadas (500 Hz → 1500 Hz)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('espectros_db_extendido.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfica 6 guardada: espectros_db_extendido.png")

# ============================================================
# 7. COMPARATIVA CSV vs BIN
# ============================================================

print("\n" + "="*60)
print("7. COMPARATIVA CSV vs BIN")
print("="*60)

try:
    datos_csv = np.loadtxt('F500.CSV', delimiter=',', skiprows=1)
    adc_csv = datos_csv[:, 2]
    adc_bin = np.fromfile('F500.bin', dtype=np.uint16)
    
    if len(adc_bin) > len(adc_csv):
        adc_bin = adc_bin[:len(adc_csv)]
    
    voltaje_csv = adc_csv * V_REF / ADC_BITS
    voltaje_bin = adc_bin * V_REF / ADC_BITS
    
    frecuencias, magnitud_csv = calcular_fft(voltaje_csv)
    _, magnitud_bin = calcular_fft(voltaje_bin)
    
    plt.figure(figsize=(12, 5))
    plt.plot(frecuencias[:1000], magnitud_csv[:1000], 'b-', linewidth=1.5, 
             label='CSV', alpha=0.8)
    plt.plot(frecuencias[:1000], magnitud_bin[:1000], 'r--', linewidth=1.5, 
             label='BIN', alpha=0.8)
    plt.axvline(500, color='g', linestyle=':', alpha=0.5, label='500 Hz')
    plt.title('Comparativa de espectros: CSV vs BIN (superpuestos)', fontsize=14)
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Magnitud', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig('espectro_csv_bin_superpuesto.png', dpi=300)
    plt.show()
    
    diferencia = np.abs(magnitud_csv - magnitud_bin)
    print(f"✅ Gráfica 7 guardada: espectro_csv_bin_superpuesto.png")
    print(f"Diferencia máxima: {np.max(diferencia):.6f}")
    print(f"Diferencia media: {np.mean(diferencia):.6f}")
    
except FileNotFoundError:
    print("⚠️ No se encontró el archivo BIN ('F500.bin') para comparar")

# ============================================================
# 8. DETECCIÓN DE FRECUENCIA DOMINANTE
# ============================================================

print("\n" + "="*60)
print("8. DETECCIÓN DE FRECUENCIA DOMINANTE")
print("="*60)

archivos = sorted(Path('.').glob('F*.CSV'), key=lambda x: int(x.stem[1:]))

resultados_freq = []
print(f"\n{'Archivo':<12} | {'Esperada (Hz)':<15} | {'Detectada (Hz)':<15} | {'Error (Hz)':<12} | {'Error (%)':<10}")
print("-" * 70)

for archivo in archivos:
    _, voltaje = cargar_datos(archivo)
    f_esperada = int(archivo.stem[1:])
    
    frecuencias, magnitud = calcular_fft(voltaje)
    
    idx_max = np.argmax(magnitud[1:int(2500 * N_FFT / FS)]) + 1
    f_detectada = frecuencias[idx_max]
    error_hz = abs(f_detectada - f_esperada)
    error_pct = (error_hz / f_esperada) * 100
    
    resultados_freq.append({
        'archivo': archivo.stem,
        'esperada': f_esperada,
        'detectada': f_detectada,
        'error_hz': error_hz,
        'error_pct': error_pct
    })
    
    print(f"{archivo.stem:<12} | {f_esperada:15d} | {f_detectada:15.1f} | {error_hz:12.3f} | {error_pct:9.3f}")

print("-" * 70)

errores = [r['error_hz'] for r in resultados_freq]
print(f"\n📊 ESTADÍSTICAS DE ERROR:")
print(f"  Error promedio:  {np.mean(errores):.4f} Hz")
print(f"  Error máximo:    {np.max(errores):.4f} Hz")
print(f"  Error mínimo:    {np.min(errores):.4f} Hz")
print(f"  Desv. estándar:  {np.std(errores):.4f} Hz")

df_freq = pd.DataFrame(resultados_freq)
df_freq.to_csv('frecuencias_dominantes.csv', index=False)
print("\n✅ Resultados guardados en 'frecuencias_dominantes.csv'")

# Gráfica de errores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

frecs = [r['esperada'] for r in resultados_freq]
err_hz = [r['error_hz'] for r in resultados_freq]
err_pct = [r['error_pct'] for r in resultados_freq]

ax1.bar(frecs, err_hz, width=60, color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Frecuencia esperada (Hz)', fontsize=12)
ax1.set_ylabel('Error absoluto (Hz)', fontsize=12)
ax1.set_title('Error en la detección de frecuencia', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(400, 1600)

for i, (f, e) in enumerate(zip(frecs, err_hz)):
    ax1.text(f, e + 0.5, f'{e:.3f}', ha='center', fontsize=9, fontweight='bold')

ax2.bar(frecs, err_pct, width=60, color='coral', edgecolor='black', alpha=0.8)
ax2.set_xlabel('Frecuencia esperada (Hz)', fontsize=12)
ax2.set_ylabel('Error (%)', fontsize=12)
ax2.set_title('Error porcentual en la detección de frecuencia', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(400, 1600)

for i, (f, e) in enumerate(zip(frecs, err_pct)):
    ax2.text(f, e + 0.05, f'{e:.3f}%', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Precisión en la detección de frecuencia dominante', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('error_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfica 8 guardada: error_frecuencias.png")

# ============================================================
# 9. PUNTOS POR CICLO VS FRECUENCIA
# ============================================================

print("\n" + "="*60)
print("9. PUNTOS POR CICLO VS FRECUENCIA")
print("="*60)

frecuencias_med = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500])
puntos_ciclo = FS / frecuencias_med

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

f_teorica = np.linspace(400, 1600, 1000)
p_teorica = FS / f_teorica

ax.plot(f_teorica, p_teorica, 'b-', linewidth=2, label='Relación teórica')

ax.scatter(frecuencias_med, puntos_ciclo, color='red', s=120, zorder=5, 
           label='Mediciones', edgecolors='darkred', linewidth=2)

for f, p in zip(frecuencias_med, puntos_ciclo):
    ax.text(f + 15, p + 0.2, f'{p:.1f}', fontsize=9, color='darkred', fontweight='bold')

ax.axhline(y=10, color='green', linestyle='--', alpha=0.7, linewidth=1.5, 
           label='Excelente (≥10 pts)')
ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
           label='Aceptable (≥5 pts)')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
           label='Límite Nyquist (2 pts)')

ax.axhspan(10, 12, alpha=0.15, color='green')
ax.axhspan(5, 10, alpha=0.15, color='yellow')
ax.axhspan(2, 5, alpha=0.15, color='orange')
ax.axhspan(0, 2, alpha=0.15, color='red')

ax.annotate('Excelente', xy=(450, 11), fontsize=10, color='green', fontweight='bold')
ax.annotate('Buena', xy=(650, 8), fontsize=10, color='darkorange', fontweight='bold')
ax.annotate('Aceptable', xy=(850, 4.5), fontsize=10, color='orange', fontweight='bold')
ax.annotate('Crítica\n(aliasing)', xy=(1350, 1.2), fontsize=10, color='red', fontweight='bold')

ax.annotate('¡Límite práctico!\n~1000 Hz (5 pts)', 
            xy=(1000, 5), xytext=(1100, 7),
            arrowprops=dict(arrowstyle='->', color='purple', linewidth=2),
            fontsize=10, color='purple', fontweight='bold')

ax.set_xlabel('Frecuencia (Hz)', fontsize=12)
ax.set_ylabel('Puntos por ciclo', fontsize=12)
ax.set_title('Relación puntos por ciclo vs frecuencia de la señal', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(400, 1600)
ax.set_ylim(0, 12)
ax.legend(loc='upper right', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('puntos_por_ciclo.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfica 9 guardada: puntos_por_ciclo.png")

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*60)
print("📊 RESUMEN DE GRÁFICAS GENERADAS")
print("="*60)
print("1.  grafica1_tiempo_offset_real.png")
print("2.  grafica2_comparativa_offset_real.png")
print("3.  offset_analysis.png")
print("4.  offset_impacto_fft.png")
print("5.  espectros_lineales.png")
print("6.  espectros_db_extendido.png")
print("7.  espectro_csv_bin_superpuesto.png")
print("8.  error_frecuencias.png")
print("9.  puntos_por_ciclo.png")
print("="*60)
print("✅ ¡Todas las gráficas generadas exitosamente!")