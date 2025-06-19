import sys
import os

# Asegurar ruta base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.run_processing import run_processing

def main():
    print("\nIniciando procesamiento de videos...\n")
    run_processing()
    print("\nProcesamiento finalizado.\n")

if __name__ == "__main__":
    main()
