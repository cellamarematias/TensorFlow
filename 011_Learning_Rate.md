# 🚀 ¿Qué es el Learning Rate (Tasa de Aprendizaje)?

El **learning rate** (o tasa de aprendizaje) es uno de los **hiperparámetros más importantes** en el entrenamiento de modelos de machine learning, especialmente en redes neuronales.

Es el valor que determina **cuánto se actualizan los pesos del modelo** en cada paso del entrenamiento cuando se minimiza la función de pérdida.

---

## 🧠 ¿Cómo funciona?

Durante el entrenamiento, el modelo ajusta sus pesos para reducir el error (pérdida). Esto se hace mediante un algoritmo de optimización (como SGD o Adam). El learning rate controla **el tamaño de estos ajustes**.

```
peso_nuevo = peso_actual - learning_rate * gradiente
```

---

## 📊 Ejemplo visual:

Imaginá que estás bajando una montaña en bicicleta (querés llegar al punto más bajo = mínimo de la pérdida):

- 🔹 Un learning rate **muy bajo**: avanzás despacio → tarda mucho en llegar.
- 🔹 Un learning rate **muy alto**: das saltos grandes → podés pasarte del mínimo o incluso **divergir**.

---

## ✅ ¿Qué valores usar?

No hay una única respuesta, pero algunos valores comunes:

- 0.01 → valor clásico y seguro
- 0.001 → más pequeño, más preciso, más lento
- 0.1 → agresivo, más rápido pero puede ser inestable

Los optimizadores modernos como `Adam` suelen funcionar bien con `learning_rate=0.001`.

---

## 🧪 ¿Dónde se define?

En Keras / TensorFlow:

```
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

En Scikit-learn (cuando aplica), se encuentra en hiperparámetros como `eta0` o `learning_rate_init`.

---

## 📌 ¿En qué modelos aplica?

- Redes neuronales (ANN, CNN, RNN, etc.)
- Regresiones y modelos lineales con optimización iterativa
- Modelos de boosting (como XGBoost o LightGBM)
- En general, cualquier modelo que use **gradiente descendente**

---

## 🔍 Buenas prácticas

- Comenzá con `0.001` o `0.01`.
- Probá usar **learning rate decay** (bajar el LR a lo largo del tiempo).
- Usá herramientas como **Learning Rate Schedulers** o **callbacks** para ajustarlo dinámicamente.
- Graficá la pérdida durante el entrenamiento para detectar si el learning rate es muy alto (oscilaciones) o muy bajo (muy lento).

---

## 📉 Tip visual para elegirlo

Una técnica útil es entrenar el modelo en una sola época con diferentes learning rates y graficar la pérdida para ver cuál valor es más estable. Esto se llama **"Learning Rate Finder"**.

---

# 🧪 Experimento Interactivo: Explorando el Learning Rate en una Red Neuronal

Este experimento está diseñado para visualizar cómo el **learning rate (tasa de aprendizaje)** afecta al entrenamiento de una red neuronal.

---

## 📦 Requisitos

Asegurate de tener TensorFlow y Matplotlib:

```
!pip install tensorflow matplotlib
```

---

## 📊 Paso 1: Dataset simple

Usaremos un dataset sintético con una relación lineal simple.

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset lineal: y = 3x + 7
X = np.linspace(-10, 10, 100)
y = 3 * X + 7 + np.random.normal(0, 2, size=X.shape)  # con algo de ruido

# Redimensionamos X para que sea una matriz columna
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
```

---

## 🧠 Paso 2: Crear función para entrenar con distinto learning rate

```
def build_and_train_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    history = model.fit(X, y, epochs=100, verbose=0)
    return history.history['loss']
```

---

## 🧪 Paso 3: Probar distintos valores de learning rate

```
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
losses = {}

for lr in learning_rates:
    print(f"Entrenando con learning rate = {lr}")
    losses[lr] = build_and_train_model(lr)
```

---

## 📈 Paso 4: Graficar resultados

```
plt.figure(figsize=(10, 6))
for lr, loss in losses.items():
    plt.plot(loss, label=f"lr={lr}")

plt.title("Pérdida durante el entrenamiento vs Learning Rate")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🧾 ¿Qué deberías observar?

- 🔹 **lr = 1e-2**: puede que sea muy rápido o inestable.
- 🔹 **lr = 1e-3**: suele ser un buen valor base.
- 🔹 **lr = 1e-4 o 1e-5**: entrenamiento más lento, pero quizás más preciso.

---

¿Querés que lo adaptemos a un dataset real como diabetes, housing, etc.?

