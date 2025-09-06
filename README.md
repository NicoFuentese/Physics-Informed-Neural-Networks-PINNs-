# ¿Que son los PINNs?
Los **PINNs** son redes neuronales que incorporan el conocimiento fisico de un sistema, normalmente expresado en **ecuaciones diferenciales parciales (PDEs)**, ecuaciones fraccionales, integro-diferenciales o estocasticas *dentro de la funcion de perdida* de la Neuronal Network (NN).

### **Construccion de una PINN:**
Una PINN combina 3 bloques principales:
1. Red Neuronal (NN) : Recibe variables de entrada y produce un valor estimado de la solucion.
2. Red Fisica (Physics - Informed) : Calcula los residuos de la PDE y condiciones de frontera usando diferenciacion automoatica.
3. Mecanismo de retroalimentacion : Ajusta los parametros de la red minimizando la funcion de perdida total: 

$$
L(\theta) = w_{F}L_{F}(\theta) + w_{B}L_{B}(\theta) + w_{D}L_{data}(\theta)
$$

Donde:

* $$L_{F}$$ : Perdida por la PDE(residuos).
* $$L_{B}$$ : Perdida por condiciones de frontera/iniciales
* $$L_{data}$$ : Error frente a datos disponibles.

# Estructura del Repositorio
* **notebooks/** : ejemplos en Python usando PyTorch/TensorFlow
* **pinn_basic.ipynb** : implementacion de un PINN sencillo para la ecuacion de Burguers 1D.
* **pinn_inverse.ipynb** : ejemplo de problema inverso con datos ruidosos.
* **utils/** : funciones auxiliares para condiciones de frontera, sampling de puntos y visualizacion.

# Referencias

* Cuomo, S., Schiano Di Cola, V., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What’s Next. Journal of Scientific Computing, 92(88). DOI: 10.1007/s10915-022-01939-z
* 