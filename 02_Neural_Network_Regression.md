# 🤖 ¿Qué es un problema de regresión (o análisis de regresión) en Deep Learning?

Un **problema de regresión** (o análisis de regresión) es como decirle a una red neuronal:  
> *"Ey, necesito que me des un número, no una etiqueta rara tipo 'gato' o 'perro'"*.

---

## 🧠 ¿Qué es un problema de regresión?

En el contexto de **deep learning** (o machine learning en general), un problema de regresión ocurre cuando **el objetivo es predecir un valor numérico continuo**.

---

## 📊 Ejemplos reales de regresión

| Problema | Qué se predice (valor continuo) |
|---------|-----------------------------|
| Predecir el precio de una casa | $123,456 |
| Estimar la temperatura para mañana | 27.3 °C |
| Predecir el ingreso de una persona según su edad y educación | $34,500 |
| Calcular la demanda de un producto el próximo mes | 10,483 unidades |

Nada de "clase A", "clase B". Solo números. Porque claro, el mundo real es más gris que blanco o negro (y más difícil de entrenar).

---

## 🔬 ¿Qué hace el modelo?

- Aprende una **función matemática** (a veces muy compleja) que relacione las entradas (features) con una **salida numérica**.
- Intenta **minimizar el error** entre sus predicciones y los valores reales.
- Usa métricas como:
  - **MAE (Mean Absolute Error)**: "¿cuánto me estoy equivocando, en promedio?"
  - **MSE (Mean Squared Error)**: "¿y si castigo más los errores grandes?"
  - **RMSE**: como MSE, pero con menos unidades raras.

---

## 🧪 ¿Cómo se ve una red de regresión?

Una red neuronal para regresión tiene estas pintas:

- Capas ocultas: ReLU, tanh, cosas así.
- Capa de salida: **una sola neurona** (porque queremos *un número*).
- Función de activación de salida: **lineal** (no queremos limitar el valor a [0,1] o cosas raras).

```
# Ejemplo simple en Keras
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)  # <--- salida de regresión
])
```

---

## 🧩 Diferencia con clasificación:

| Clasificación | Regresión |
|---------------|-----------|
| Salida: categoría | Salida: número |
| Ej: gato, perro | Ej: 42.7, 103.4 |
| Métricas: accuracy | Métricas: MAE, MSE, RMSE |
| Activación final: softmax/sigmoid | Activación final: lineal |

---

