# Predicción del rendimiento académico

Proyecto académico de aprendizaje automático que explora el rendimiento de estudiantes mediante clasificación, regresión y agrupamiento.

## Qué contiene

| Módulo | Contenido |
|---|---|
| [Exploración y preprocesado](notebooks/01_EDA_preprocesado.py) | Análisis descriptivo y preparación de variables |
| [Clasificación](notebooks/02_clasificacion.py) | Comparación de modelos supervisados |
| [Regresión](notebooks/03_regresion.py) | Predicción de una variable numérica y análisis de importancia |
| [Aprendizaje no supervisado](notebooks/04_no_supervisado.py) | PCA y análisis de grupos |

## Preparación

Usa Python 3.9 o superior y crea un entorno virtual:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Coloca el conjunto de datos autorizado en `data/rendimiento_estudiantes.csv`. Los scripts esperan el formato y las columnas del proyecto original; los datos no se distribuyen en este repositorio. Consulta [las indicaciones sobre datos](data/README.md).

Desde la raíz del proyecto:

```bash
python main.py
```

Los resultados se generan en `outputs/`.

## Estado y alcance

Código académico recuperado y organizado para consulta. Se ha comprobado la sintaxis de los archivos Python; la ejecución completa requiere los datos y las dependencias y no se ha validado en esta reorganización. Las dependencias se enumeran sin fijar versiones porque no se dispone de un entorno original verificado. El lanzador original captura errores por módulo: revisa la salida de cada etapa aunque aparezca un mensaje final de ejecución completada.

La [documentación original](docs/README-original.md) se conserva como referencia. No se atribuyen métricas ni resultados que no se hayan reproducido.

## Otros trabajos

[Prácticas de iMat](https://github.com/psoto212/imat-practicas) · [Análisis de reseñas](https://github.com/psoto212/analisis-resenas-multibase)
