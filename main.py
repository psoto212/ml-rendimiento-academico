"""
=============================================================
PROYECTO FINAL - MACHINE LEARNING
Predicción del Éxito Académico en Educación Superior
=============================================================
main.py – Script principal para ejecutar todo el proyecto
Ejecución: python main.py
"""

import os
import sys
import time

print("=" * 70)
print(" PROYECTO FINAL – MACHINE LEARNING")
print(" Predicción del Éxito Académico en Educación Superior")
print("=" * 70)
print()

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')

scripts = [
    ('01_EDA_preprocesado.py',  '1. Exploración y Análisis Descriptivo (EDA)'),
    ('02_clasificacion.py',     '2. Clasificación Multiclase'),
    ('03_regresion.py',         '3. Regresión – Nota Media 2º Semestre'),
    ('04_no_supervisado.py',    '4. Aprendizaje No Supervisado'),
]

total_start = time.time()

for script_file, description in scripts:
    script_path = os.path.join(NOTEBOOKS_DIR, script_file)
    print()
    print("─" * 70)
    print(f"▶ Ejecutando: {description}")
    print("─" * 70)
    t0 = time.time()

    # Cambiamos el directorio al de notebooks para que __file__ funcione
    original_dir = os.getcwd()
    os.chdir(NOTEBOOKS_DIR)

    with open(script_file, 'r', encoding='utf-8') as f:
        code = f.read()

    try:
        exec(compile(code, script_file, 'exec'), {'__file__': script_file})
        elapsed = time.time() - t0
        print(f"\n✓ Completado en {elapsed:.1f}s")
    except Exception as e:
        print(f"\n✗ Error en {script_file}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_dir)

total_elapsed = time.time() - total_start
print()
print("=" * 70)
print(f"✓ PROYECTO COMPLETADO en {total_elapsed:.1f}s")
print()
print("Archivos generados en outputs/:")
outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
if os.path.exists(outputs_dir):
    for f in sorted(os.listdir(outputs_dir)):
        size = os.path.getsize(os.path.join(outputs_dir, f))
        print(f"  {f:50s}  ({size/1024:.1f} KB)")
print("=" * 70)
