# Optimización No Lineal: Métodos Irrestrictos aplicados a SINGX y ALMOST

Este repositorio contiene la implementación en Python de cuatro métodos de optimización irrestricta aplicados a dos funciones clásicas de prueba: **SINGX** (Powell singular extendida) y **ALMOST** (Brown Almost Linear). Forma parte de la Tarea de NLP del curso de Modelos de Optimización.

## Estructura del repositorio

- **`funciones.py`**: contiene las definiciones de `singx_f`, `singx_grad`, `singx_hess`, `almost_f`, `almost_grad`, `almost_hess` y los cuatro métodos (`max_descenso`, `newton_method`, `bfgs`, `trust_region_cauchy`). Todos los notebooks importan desde este módulo, de modo que el código no se repite.
- **`01_max_descenso/`**: notebooks del método de máximo descenso con búsqueda lineal de Armijo.
- **`02_newton/`**: notebooks del método de Newton (con resolución del sistema lineal).
- **`03_cuasi_newton/`**: notebooks del método BFGS con búsqueda lineal de Wolfe.
- **`04_region_confianza/`**: notebooks del método de región de confianza con paso de Cauchy.

Cada notebook contiene: la descripción del método, la importación desde `funciones.py`, la ejecución sobre la función correspondiente, los resultados esperados comentados en el código y el análisis del mínimo encontrado.

## Requerimientos

- **Python 3.9 o superior**
- **NumPy** (probado con 1.24+)
- **Jupyter Notebook** o **Jupyter Lab** (o VS Code con la extensión de Jupyter)

Para instalar las dependencias, ejecutar en la terminal:

```bash
pip install numpy jupyter

