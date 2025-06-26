import numpy as np

# Usa raw string o doble barra
buffer_path = r"C:\Users\sebas\OneDrive\Desktop\SISTEMA DE CAIDAS\datasets\procesados\test\caida\benchmark_caida_fmaskb3_B_N_06_buffer_0.npy"

buffer = np.load(buffer_path)

print("Forma del buffer:", buffer.shape)
print("Tipo de datos:", buffer.dtype)
