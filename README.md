# House Prices MLOps

## 1. Descripción del proyecto

Proyecto de la actividad **TC5061 — Experimentación y seguimiento con MLflow**. Implementa un flujo reproducible de aprendizaje automático para predecir `SalePrice`, el precio de venta de viviendas del conjunto Ames Housing utilizado en House Prices de Kaggle.

El flujo comprende Data Understanding, Data Preparation, Modeling, validación cruzada, experimentación parametrizable y seguimiento con MLflow. El modelo base de la actividad es **Ridge Regression**. Los notebooks se utilizan para exploración y análisis; la ejecución del experimento se realiza mediante `scripts/train.py`.

## 2. Estructura y responsabilidades

```text
house-prices-mlops/
├── configs/
│   └── variable_types.json
├── data/
│   ├── raw/
│   │   └── train.csv
│   ├── external/
│   │   └── data_description.txt
│   └── interim/
│       ├── split_manifest.csv
│       └── tracking/
├── house_prices_mlops/
│   ├── dataset.py
│   ├── pipeline.py
│   ├── metadata.py
│   ├── data_tracker.py
│   ├── tracking/
│   ├── preparation/
│   │   ├── structure.py
│   │   ├── missing.py
│   │   └── feature_engineering.py
│   └── modeling/
│       ├── pipelines.py
│       ├── preprocessing.py
│       ├── distributions.py
│       ├── feature_selection.py
│       ├── encoding.py
│       └── scaling.py
├── scripts/
│   └── train.py
├── notebooks/
│   ├── 00_house_prices_original.ipynb
│   ├── 01_data_understanding/
│   │   ├── 01_structure.ipynb
│   │   ├── 02_quality.ipynb
│   │   ├── 03_target_analysis.ipynb
│   │   └── 04_feature_analysis.ipynb
│   ├── 02_data_preparation/
│   │   ├── 01_structure.ipynb
│   │   ├── 02_missing.ipynb
│   │   ├── 03_feature_engineering.ipynb
│   │   ├── 04_distributions.ipynb
│   │   ├── 05_outliers.ipynb
│   │   ├── 06_feature_selection.ipynb
│   │   ├── 07_encoding.ipynb
│   │   └── 08_scaling.ipynb
│   └── 03_modeling/
│       └── 01_experiments.ipynb
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

El árbol muestra los elementos principales; no enumera todos los archivos de la plantilla.

- **`dataset.py`:** carga los datos originales, crea o valida la partición train/test y separa predictores y objetivo. Conserva `Id` para trazabilidad, pero lo excluye de `X` junto con `SalePrice`.
- **`pipeline.py`:** reconstruye estados de Data Preparation en memoria: `structure`, `missing` y `common`. Cada llamada aplica las etapas necesarias desde la partición original, sin guardar datasets transformados intermedios.
- **`preparation/`:** contiene las funciones reutilizables de representación, ausencias semánticas e ingeniería de variables comunes.
- **`modeling/`:** contiene selección de variables, distribuciones, Encoding, Scaling y composición del pipeline experimental. `preprocessing.py` construye las ramas; `pipelines.py` conecta los componentes.
- **`scripts/train.py`:** carga entrenamiento, construye Ridge, evalúa cinco folds, ajusta el pipeline final y registra la corrida en MLflow.
- **`notebooks/`:** documenta investigaciones y decisiones. `01_experiments.ipynb` analiza datos y consulta resultados registrados; no duplica el entrenamiento.
- **`DataTracker` y `metadata.py`:** documentan hallazgos, planes y decisiones, y gestionan las clasificaciones conceptuales. El pipeline de Modeling no ejecuta automáticamente los planes JSON.

## 3. Requisitos e instalación

El proyecto declara **Python 3.10** mediante `requires-python = "~=3.10.0"`. Las cinco corridas de la actividad utilizaron **Python 3.10.22**.

Las dependencias se administran con **`pyproject.toml` y `uv.lock`**. No se utiliza un `requirements.txt` como mecanismo de instalación del proyecto. Entre las dependencias principales están scikit-learn, MLflow, NumPy, pandas, SciPy, Jupyter, matplotlib, seaborn y statsmodels; estas últimas herramientas apoyan la exploración y los análisis estadísticos.

Versiones registradas en las corridas de la actividad:

| Dependencia | Versión |
|---|---:|
| MLflow | 3.17.0 |
| scikit-learn | 1.7.2 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| cloudpickle | 3.1.2 |
| joblib | 1.6.0 |

Con Python 3.10 y `uv` disponibles, desde la raíz del proyecto:

```bash
uv sync --locked
```

El comando sincroniza `.venv` con el archivo de bloqueo. En la terminal Bash del contenedor de trabajo, activa ese entorno:

```bash
source .venv/bin/activate
python --version
```

Antes de entrenar, debe existir **`data/raw/train.csv`**, con las columnas `Id` y `SalePrice`. El script no descarga el dataset. Para los análisis de variables se utiliza `data/external/data_description.txt` y `configs/variable_types.json`.

La configuración actual ejecuta el código en el contenedor **`kaggle_practice`**, conectado a la misma red que MLflow. La instalación de dependencias no inicia esos servicios; su preparación se describe en la sección 6.

## 4. Ejecución del entrenamiento

Desde la raíz del proyecto, dentro del entorno y la red de contenedores indicados:

```bash
python scripts/train.py
```

Ejemplo parametrizado en Bash:

```bash
python scripts/train.py \
    --alpha 10 \
    --tol 0.0001 \
    --solver auto \
    --seed 42
```

La misma ejecución en una sola línea, también apropiada para PowerShell:

```text
python scripts/train.py --alpha 10 --tol 0.0001 --solver auto --seed 42
```

Con `uv`, sin activar manualmente el entorno:

```bash
uv run --locked python scripts/train.py --alpha 10 --tol 0.0001 --solver auto --seed 42
```

Los argumentos existentes son:

| Argumento | Tipo | Predeterminado | Significado y validación |
|---|---|---:|---|
| `--alpha` | float | `1.0` | Intensidad de regularización; debe ser finita y mayor o igual a cero. |
| `--tol` | float | `0.0001` | Tolerancia del solver; debe ser finita y mayor que cero. |
| `--solver` | string | `auto` | Opciones: `auto`, `svd`, `cholesky`, `lsqr`, `sparse_cg`, `sag`, `saga`. |
| `--seed` | int | `42` | Semilla; debe estar entre 0 y 4294967295. |

`lbfgs` no está habilitado porque requiere `positive=True`, que no pertenece a este experimento. La influencia de `tol` depende del solver seleccionado.

Desde el equipo anfitrión también puede invocarse el entorno existente directamente:

```text
docker exec -w /home/jovyan/notebooks/house_prices_cookiecutter/house-prices-mlops kaggle_practice .venv/bin/python scripts/train.py --alpha 10 --tol 0.0001 --solver auto --seed 42
```

El script imprime las métricas por fold, el resumen, el identificador de corrida, la URI del modelo y el enlace de MLflow.

## 5. Flujo de entrenamiento y evaluación

1. **Carga:** `load_stage("common")` recupera entrenamiento y ejecuta Structure, Missing e ingeniería de variables comunes.
2. **Separación:** `get_predictors_and_target()` devuelve `X` sin `Id` ni `SalePrice`, y `y` en la escala original del precio.
3. **Construcción:** `build_model_pipeline()` recibe `Ridge(alpha, tol, solver, random_state=seed)`, con Feature Selection desactivado, `distribution_transformations={}`, StandardScaler y `log_target=False`.
4. **Validación cruzada:** se utiliza `KFold(n_splits=5, shuffle=True, random_state=seed)` únicamente sobre entrenamiento.
5. **Evaluación:** se calculan MAE, RMSE y R² por fold, sus promedios y desviaciones estándar con `ddof=0`. MAE y RMSE se presentan positivos en la escala original de `SalePrice`.
6. **Ajuste final:** dentro de la corrida de MLflow, el pipeline completo se ajusta con todos los datos de entrenamiento disponibles.
7. **Registro:** se conservan parámetros, métricas, artefactos y el pipeline final entrenado.

Cada fold ajusta sus propias medianas de imputación numérica, categorías de One-Hot, estadísticas de StandardScaler y coeficientes de Ridge. Los órdenes ordinales son explícitos; las categorías desconocidas y los faltantes tienen códigos diferenciados. No se aplican transformaciones de distribución en este experimento.

La partición determinista original es de **80 % entrenamiento y 20 % prueba**, con orden estable por `Id` y semilla 42. Para el dataset actual son **1,168 viviendas de entrenamiento y 292 de prueba**; `common` proporciona **91 predictores** antes del Encoding. El manifiesto `data/interim/split_manifest.csv` conserva la asignación y una huella del dataset. Si los datos cambian, la validación rechaza el manifiesto en lugar de sobrescribirlo silenciosamente.

**El conjunto de prueba reservado no participa en la validación cruzada ni en el ajuste final del script.** Las desviaciones entre folds describen variabilidad de evaluación; no son intervalos de confianza.

## 6. Seguimiento con MLflow

El script utiliza actualmente el experimento **`house-prices-ridge-cookiecutter`**, de acuerdo con la configuración existente en `scripts/train.py`.

Las direcciones cumplen funciones diferentes:

| Uso | Dirección |
|---|---|
| Tracking desde el contenedor de trabajo | `http://mlflow:5000` |
| Interfaz web desde el navegador del equipo anfitrión | [http://localhost:5000](http://localhost:5000) |
| Experimento de las cinco corridas | [house-prices-ridge-cookiecutter](http://localhost:5000/#/experiments/8) |

`mlflow` es el alias interno del contenedor `mlflow_server`; no es necesariamente un nombre resoluble desde el anfitrión. El puerto 5000 está publicado en el anfitrión. Ejecutar el script o instalar sus dependencias **no inicia automáticamente MLflow**.

La infraestructura existente se administra fuera de este repositorio mediante `X:\Repositorios\Personal\Kaggle\docker-compose.yml`, con los servicios `mlflow`, `minio` y `ds_workspace`. En la disposición actual, desde la raíz del repositorio **en el equipo anfitrión**, pueden iniciarse MLflow y su almacenamiento con:

```text
docker compose -f ../../../docker-compose.yml up -d mlflow minio
```

Este comando depende del archivo Compose externo existente; no hay un servidor ni un archivo Compose nuevo dentro de este proyecto. Para usar `docker exec`, el contenedor de trabajo `kaggle_practice` también debe estar iniciado; corresponde al servicio `ds_workspace` de esa infraestructura.

Cada corrida registra:

- **Parámetros:** `alpha`, `tol`, `solver`, `seed`, configuración del pipeline, validación cruzada, Encoding, tamaño de entrenamiento y versiones reales del entorno.
- **Métricas:** `mae_mean`, `mae_std`, `rmse_mean`, `rmse_std`, `r2_mean` y `r2_std`; además, `fold_mae`, `fold_rmse` y `fold_r2`, con el número de fold como paso.
- **`cv_fold_metrics.csv`:** resultados individuales de las cinco particiones.
- **`cv_summary.json`:** promedios, desviaciones, `std_ddof` y versiones utilizadas.
- **Modelo:** pipeline completo ajustado, registrado mediante `mlflow.sklearn.log_model()`, con firma, código de sus componentes y versiones de dependencias. Se serializa con `cloudpickle` y su URI se imprime al finalizar.

Los CSV y JSON de métricas se generan en un directorio temporal y se envían como artefactos. No son datasets transformados ni estados adicionales de Data Preparation.

## 7. Configuraciones y resultados de la actividad

Las etiquetas E01–E05 identifican en este documento las cinco configuraciones verificadas en MLflow. Las corridas comparten el nombre registrado `ridge_baseline_standard_no_selection_no_distributions`; los enlaces permiten identificar cada ejecución sin ambigüedad.

| Experimento | alpha | tol | solver | seed |
|---|---:|---:|---|---:|
| [E01](http://localhost:5000/#/experiments/8/runs/43394220da374418811c09de22f1365e) | 1.0 | 0.0001 | auto | 42 |
| [E02](http://localhost:5000/#/experiments/8/runs/77986c23f7f041fb9beed2c5515c9249) | 0.1 | 0.0001 | auto | 42 |
| **[E03](http://localhost:5000/#/experiments/8/runs/ce73fbb87d6b4af2a47d088268b5bf98)** | **10.0** | **0.0001** | **auto** | **42** |
| [E04](http://localhost:5000/#/experiments/8/runs/bffc63b4e1634c4385b2d2957b46aefb) | 1.0 | 0.001 | auto | 42 |
| [E05](http://localhost:5000/#/experiments/8/runs/b0d3a46ccc1a4b16ac063f719b390318) | 1.0 | 0.0001 | lsqr | 42 |

Resultados de validación cruzada de cinco folds, expresados como **promedio ± desviación estándar**:

| Experimento | MAE | RMSE | R² |
|---|---:|---:|---:|
| E01 | 19,829.01 ± 1,587.01 | 37,105.19 ± 11,831.55 | 0.754848 ± 0.144928 |
| E02 | 20,614.13 ± 2,306.12 | 47,918.37 ± 21,329.63 | 0.570076 ± 0.308692 |
| **E03** | **19,523.97 ± 1,453.21** | **36,105.62 ± 10,438.73** | **0.772113 ± 0.115139** |
| E04 | 19,829.01 ± 1,587.01 | 37,105.19 ± 11,831.55 | 0.754848 ± 0.144928 |
| E05 | 19,823.86 ± 1,594.42 | 37,056.48 ± 11,817.29 | 0.755424 ± 0.144896 |

**E03 es la mejor configuración de las cinco evaluadas:** presenta el menor MAE promedio, el menor RMSE promedio y el mayor R² promedio. Es una conclusión sobre estas configuraciones y esta validación cruzada, no una evaluación final sobre el conjunto de prueba reservado.

Para repetir las cinco configuraciones desde el entorno preparado:

```bash
python scripts/train.py --alpha 1 --tol 0.0001 --solver auto --seed 42
python scripts/train.py --alpha 0.1 --tol 0.0001 --solver auto --seed 42
python scripts/train.py --alpha 10 --tol 0.0001 --solver auto --seed 42
python scripts/train.py --alpha 1 --tol 0.001 --solver auto --seed 42
python scripts/train.py --alpha 1 --tol 0.0001 --solver lsqr --seed 42
```

## 8. Reproducibilidad

`--seed` fija las semillas de NumPy y del módulo `random`, y se entrega a Ridge y a KFold. Mantener la semilla 42, los mismos datos y el mismo orden de filas conserva las particiones de validación entre los cinco experimentos, lo que permite comparar sus configuraciones sobre las mismas viviendas. La partición train/test tiene su propia semilla fija 42 y permanece definida por el manifiesto, incluso si se cambia la semilla de validación cruzada.

`uv.lock` administra las versiones resueltas del entorno. Cada corrida registra las versiones realmente utilizadas y el modelo incorpora requisitos con versiones explícitas. Para repetir una configuración, utilice el mismo dataset, manifiesto, código y argumentos; sincronice las dependencias con `uv sync --locked`.

Se comprobaron dos ejecuciones previas de la configuración base, en el mismo entorno, comparando tanto las métricas por fold como los seis resúmenes registrados, con tolerancia relativa `1e-10` y absoluta `1e-7`. Los resultados coincidieron dentro de esa tolerancia. Esta comprobación no implica identidad numérica garantizada entre sistemas operativos, versiones o implementaciones de álgebra lineal diferentes.

Los identificadores de corrida y los timestamps sirven para localizar ejecuciones, pero no determinan la reproducibilidad. Cada nueva ejecución crea una corrida independiente en MLflow; no sobrescribe las métricas de una corrida anterior.
