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

### Modelamiento matematico-fisico de PINNs:

#### <ins>Problema fisico general:</ins>
Muchos problemas fisicos se describen por **ecuaciones diferenciales**. El caso mas general es:

$$
F(u(z);\gamma) = f(z), \qquad z\in\Omega
$$

con condiciones de borde:

$$
B(u(z)) = g(z), \qquad z\in\partial\Omega
$$

Donde:
* $$u(z)$$ = solucion desconocida (ejemplo: Temperatura, Presion, etc.)
* $$z = [x_1, x_2, ..., x_{d-1},t]$$ = coordenadas espacio-temporales dentro del dominio $$\omega$$
* $$F$$ = operador diferencial (derivadas parciales de orden superior).
* $$\gamma$$ = parametros fisicos del sistema.
* $$\Omega\subset\mathbb{R^d}$$ = dominio.
* $$\partial\Omega$$  frontera del dominio.

#### <ins>Aproximacion con una Neural Network:</ins>
Se entrena una red neuronal $$u_{\theta}(z)$$ con parametros $$\theta$$ (pesos y sesgos) para aproximar la solucion:

$$
u_{\theta}(z) \approx u(z)
$$

#### <ins>Residuos (PDE + Condiciones de frontera):</ins>
La red no solo se ajusta a datos, sino tambien a las leyes fisicas. Considerar que si $$u_{\theta}$$ fuera la solucion exacta, entoncs $$r_{F} = 0$$ y $$r_{B} = 0$$.

* Residuo de la PDE:

$$
r_{F}\[u_{\theta}\](z) = F(u_{\theta}(z);\gamma) - f(z)
$$

* Residuo de condiciones de frontera:

$$
r_{B}[u_{\theta}](z) = B(u_{\theta}(z):\gamma) - g(z)
$$

#### <ins>Funcion de perdida:</ins>
La PINN minimiza una **perdida compuesta:**

$$
L(\theta) = w_{F}L_{F}(\theta) + w_{B}L_{B}(\theta) + w_{D}L_{data}(\theta)
$$

Donde:
* $$L_{F}(\theta)$$ : mide el error de la PDE en los puntos $$z_{i}$$

$$
L_{F}(\theta) = \frac{1}{N_c} \sum_{i=1}^{N_c} \left\| r_F[u_{\theta}](z_i)\right\|^2
$$

* $$L_{B}(\theta)$$ : mide el error en condiciones de frontera/borde
  
$$
L_{F}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \left\| r_B[u_{\theta}](z_i)\right\|^2
$$

* $$L_{data}(\theta)$$ : error con respecto a datos observados (si existen)

$$
L_{F}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \left\| r_{\theta}(z_i) - u^*(z_i) \right\|^2
$$

#### <ins>Optimizacion</ins>
Se resuelve:

$$
\theta^* = arg min_{\theta} L(\theta)
$$

Convirtiendo el **problema de resolver una PDE en un problema de optimizacion.** Se usan metodos como Adam y L-BFGS para minimizar la perdida.

# Estructura del Repositorio
* **notebooks/** : ejemplos en Python usando PyTorch/TensorFlow
* **pinn_basic.ipynb** : implementacion de un PINN sencillo para la ecuacion de Burguers 1D.
* **pinn_inverse.ipynb** : ejemplo de problema inverso con datos ruidosos.
* **utils/** : funciones auxiliares para condiciones de frontera, sampling de puntos y visualizacion.

# Referencias

* Cuomo, S., Schiano Di Cola, V., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What’s Next. Journal of Scientific Computing, 92(88). DOI: 10.1007/s10915-022-01939-z
* 